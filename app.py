import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from werkzeug.utils import secure_filename
import tempfile
import base64
import depthai as dai
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Asegurarse de que existe el directorio de uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Fijos de preprocesado
BLUR_KSIZE = (5,5)
THRESH_METHOD = cv2.THRESH_BINARY + cv2.THRESH_OTSU

# Variable global para la cámara
camera = None

def init_camera():
    global camera
    if camera is None:
        try:
            # Crear pipeline
            pipeline = dai.Pipeline()
            
            # Definir fuente - cámara RGB
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(1280, 720)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            
            # Configurar la cámara para mejor calidad
            cam_rgb.setFps(30)
            cam_rgb.initialControl.setAutoExposureEnable()
            cam_rgb.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            
            # Crear salida sin buffer
            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            xout_rgb.input.setBlocking(False)
            xout_rgb.input.setQueueSize(1)  # Solo mantener 1 frame en la cola
            cam_rgb.preview.link(xout_rgb.input)
            
            # Conectar al dispositivo
            device = dai.Device(pipeline)
            camera = device
            return True
        except Exception as e:
            print(f"Error inicializando la cámara: {e}")
            return False
    return True

def capture_photo():
    if not init_camera():
        return None, "Error al inicializar la cámara"
    
    try:
        # Obtener el stream de la cámara sin buffer
        q_rgb = camera.getOutputQueue("rgb", maxSize=1, blocking=False)
        
        # Limpiar cualquier frame antiguo en la cola
        while q_rgb.has():
            q_rgb.get()
        
        # Capturar un único frame nuevo
        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()
        
        # Verificar que el frame es válido
        if frame is None or frame.size == 0:
            return None, "Error: Frame no válido"
            
        # Limpiar imágenes antiguas
        for old_file in os.listdir(app.config['UPLOAD_FOLDER']):
            if old_file.startswith('capture_'):
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], old_file))
                except:
                    pass
        
        # Guardar la imagen
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, frame)
        
        return filename, None
    except Exception as e:
        return None, f"Error capturando foto: {str(e)}"

@app.route('/capture', methods=['POST'])
def capture():
    filename, error = capture_photo()
    if error:
        return jsonify({'error': error}), 500
    return jsonify({'filename': filename})

def ensure_odd(x):
    return x if x % 2 == 1 else max(1, x-1)

def process_image(image_path, roi, params):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error leyendo la imagen"

    # Convertir los valores del ROI a enteros
    x = int(roi[0])
    y = int(roi[1])
    w = int(roi[2])
    h = int(roi[3])

    # Verificar que los valores son válidos
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None, "ROI inválido"
    if x + w > img.shape[1] or y + h > img.shape[0]:
        return None, "ROI fuera de los límites de la imagen"

    roi_color = img[y:y+h, x:x+w]

    # Procesado
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 1)
    _, mask = cv2.threshold(blur, 0, 255, THRESH_METHOD)
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    ksz = ensure_odd(params['morph_ksize'])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    defects_mask = cv2.bitwise_and(closed, cv2.bitwise_not(mask))

    # Detectar muescas
    conts, _ = cv2.findContours(defects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    salida = roi_color.copy()
    defect_count = 0
    for c in conts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < params['min_def_area'] or bh < bw * params['asp_ratio']:
            continue
        cv2.rectangle(salida, (bx,by), (bx+bw,by+bh), (0,255,0), 2)
        defect_count += 1

    # Overlay máscara
    mask_bgr = np.zeros_like(roi_color)
    mask_bgr[:,:,0] = mask
    overlay = cv2.addWeighted(roi_color, 0.7, mask_bgr, 0.3, 0)

    # Crear mosaico
    def to_bgr(im):
        return im if im.ndim==3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    
    i1 = to_bgr(roi_color)
    i2 = to_bgr(mask)
    i3 = overlay
    i4 = salida
    
    i1 = cv2.resize(i1, (w, h), interpolation=cv2.INTER_NEAREST)
    i2 = cv2.resize(i2, (w, h), interpolation=cv2.INTER_NEAREST)
    i3 = cv2.resize(i3, (w, h), interpolation=cv2.INTER_NEAREST)
    i4 = cv2.resize(i4, (w, h), interpolation=cv2.INTER_NEAREST)
    
    top = np.hstack([i1, i2])
    bot = np.hstack([i3, i4])
    mosaic = np.vstack([top, bot])

    # Añadir status box
    status = "OK" if defect_count == 0 else "NOK"
    color = (0,255,0) if defect_count == 0 else (0,0,255)
    cv2.rectangle(mosaic, (10,10), (160,60), (0,0,0), -1)
    cv2.putText(mosaic, status, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Convertir a base64
    _, buffer = cv2.imencode('.png', mosaic)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64, status

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'filename': filename})

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    filename = data.get('filename')
    roi = data.get('roi')
    params = data.get('params')

    if not all([filename, roi, params]):
        return jsonify({'error': 'Missing parameters'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    img_base64, status = process_image(filepath, roi, params)
    if img_base64 is None:
        return jsonify({'error': status}), 500

    return jsonify({
        'image': img_base64,
        'status': status
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True) 