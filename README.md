# Aplicación Web de Detección de Holgura

Esta aplicación web permite detectar holguras en imágenes utilizando procesamiento de imagen con OpenCV. La interfaz web permite seleccionar una región de interés (ROI) y ajustar los parámetros de detección en tiempo real.

## Requisitos

- Python 3.8 o superior
- OpenCV
- Flask
- NumPy
- Pillow

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Crear un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Iniciar el servidor:
```bash
python app.py
```

2. Abrir un navegador web y acceder a:
```
http://localhost:5000
```

3. Instrucciones de uso:
   - Hacer clic en "Seleccionar imagen" para cargar una imagen
   - Usar el ratón para seleccionar la región de interés (ROI) en la imagen
   - Ajustar los parámetros usando los controles deslizantes:
     - MIN_DEF_AREA: Área mínima para considerar un defecto
     - ASP_RATIO: Relación de aspecto mínima para considerar un defecto
     - MORPH_KSIZE: Tamaño del kernel para operaciones morfológicas
   - El resultado se mostrará automáticamente debajo de la imagen original
   - El estado (OK/NOK) se mostrará en la parte derecha

## Estructura del Proyecto

```
.
├── app.py              # Aplicación principal Flask
├── requirements.txt    # Dependencias del proyecto
├── templates/         # Plantillas HTML
│   └── index.html     # Interfaz de usuario
└── uploads/          # Directorio para imágenes subidas
```

## Notas

- Las imágenes subidas se almacenan temporalmente en el directorio `uploads/`
- El tamaño máximo de archivo permitido es de 16MB
- La aplicación está optimizada para imágenes en formato PNG o JPG 