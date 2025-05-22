import os
import sys
import subprocess
from pathlib import Path

# Contenido de los archivos
ARCHIVOS = {
    'config/config_camara.yaml': """ip: "172.19.68.143"
usuario: "mjpeg"
contraseña: "mjpeg2000"
resolucion: "2688x1520"
fps: 5
url_stream: "rtsp://mjpeg:mjpeg2000@172.19.68.143:554/cam/realmonitor?channel=1&subtype=0"
altura_camara: 3.0  # metros sobre el piso
angulo_inclinacion: 30  # grados hacia abajo
altura_palet: 1.0  # altura estimada inicial del palet en metros
node_red_url: "http://localhost:1880/endpoint"  # URL de tu servidor Node-RED
""",

    'config/config_kuka.yaml': """ip: "172.19.68.144"
puerto: 7000
z_por_defecto: 1200.0  # Altura predeterminada en mm
velocidad_maxima: 50   # Velocidad porcentual (0-100)
""",

    'src/sistema_vision.py': """import cv2
import numpy as np
from deteccion_palets import DetectorPalets
from comunicacion import EnviadorCoordenadas
from utils import cargar_configuracion, transformar_coordenadas

class SistemaVisionPalets:
    def __init__(self):
        self.config = cargar_configuracion()
        self.detector = DetectorPalets()
        self.enviador = EnviadorCoordenadas(self.config['node_red_url'])
        self.matriz_homografia = np.load('data/calibracion/homografia.npy')
        self.factor_correccion_altura = self._calcular_factor_correccion()

    def _calcular_factor_correccion(self, altura_palet=None):
        \"\"\"Ajusta las coordenadas según la altura del palet\"\"\"
        altura_palet = altura_palet or self.config['altura_palet']
        h_cam = self.config['altura_camara']
        angulo = np.radians(self.config['angulo_inclinacion'])
        distancia_teorica = h_cam / np.tan(angulo)
        distancia_real = (h_cam - altura_palet) / np.tan(angulo)
        return distancia_real / distancia_teorica

    def procesar_frame(self, frame):
        palet = self.detector.detectar_palet(frame)
        
        if palet:
            centro_x, centro_y = palet['centro']
            ancho, alto = palet['dimensiones']
            
            x, y = transformar_coordenadas(centro_x, centro_y, self.matriz_homografia)
            x *= self.factor_correccion_altura
            y *= self.factor_correccion_altura
            
            datos = {
                'x': float(x),
                'y': float(y),
                'z': float(self.config['altura_palet']),
                'ancho': float(ancho),
                'alto': float(alto),
                'unidades': 'mm'
            }
            
            self.enviador.enviar_datos(datos)
            return datos
        return None

    def ejecutar(self):
        cap = cv2.VideoCapture(self.config['url_stream'])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            datos = self.procesar_frame(frame)
            self._mostrar_resultados(frame, datos)
            
            if cv2.waitKey(1) == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def _mostrar_resultados(self, frame, datos):
        if datos:
            texto = f"X: {datos['x']:.1f}mm, Y: {datos['y']:.1f}mm"
            cv2.putText(frame, texto, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Monitoreo Palets", frame)

if __name__ == "__main__":
    sistema = SistemaVisionPalets()
    sistema.ejecutar()
""",

    'src/deteccion_palets.py': """import cv2
import numpy as np

class DetectorPalets:
    def __init__(self, ancho_palet_estandar=1200, alto_palet_estandar=144):
        self.ancho_estandar = ancho_palet_estandar  # mm
        self.alto_estandar = alto_palet_estandar    # mm

    def detectar_palet(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(blurred, 50, 150)
        
        contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contornos:
            if cv2.contourArea(cnt) < 5000:
                continue
                
            perimetro = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimetro, True)
            
            if len(approx) >= 4:
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angulo = rect
                
                relacion_aspecto = max(w, h) / min(w, h)
                if 2.5 < relacion_aspecto < 4.0:
                    caja = cv2.boxPoints(rect)
                    caja = np.int0(caja)
                    
                    return {
                        'centro': (x, y),
                        'dimensiones': (w, h),
                        'angulo': angulo,
                        'contorno': caja
                    }
        return None
""",

    'src/comunicacion.py': """import requests
import json

class EnviadorCoordenadas:
    def __init__(self, url_node_red):
        self.url = url_node_red
        self.headers = {'Content-Type': 'application/json'}

    def enviar_datos(self, datos):
        try:
            respuesta = requests.post(
                self.url,
                data=json.dumps(datos),
                headers=self.headers
            )
            if respuesta.status_code != 200:
                print(f"Error al enviar: {respuesta.text}")
        except Exception as e:
            print(f"Error de conexión: {str(e)}")
""",

    'src/utils.py': """import yaml
import numpy as np
import cv2

def cargar_configuracion():
    with open('config/config_camara.yaml') as f:
        config = yaml.safe_load(f)
    return config

def transformar_coordenadas(x, y, matriz_h):
    punto = np.array([[[x, y]]], dtype=np.float32)
    transformado = cv2.perspectiveTransform(punto, matriz_h)[0][0]
    return transformado[0], transformado[1]

def calcular_factor_altura(altura_camara, angulo_inclinacion, altura_palet):
    \"\"\"Calcula factor de corrección para diferentes alturas de palet\"\"\"
    h_cam = altura_camara
    ang_rad = np.radians(angulo_inclinacion)
    return (h_cam - altura_palet) / (h_cam * np.tan(ang_rad))
""",

    'src/__init__.py': "",

    'data/calibracion/homografia.npy': "",  # Archivo vacío, se generará con calibracion.py

    'requirements.txt': """opencv-python>=4.5.0
numpy>=1.20.0
requests>=2.25.0
pyyaml>=5.4.0
"""
}

def crear_estructura():
    # Crear directorios
    directorios = [
        'config',
        'src',
        'data/calibracion',
        'data/modelos'
    ]
    
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
        print(f"Directorio creado: {directorio}")

    # Crear archivos
    for archivo, contenido in ARCHIVOS.items():
        with open(archivo, 'w') as f:
            f.write(contenido)
        print(f"Archivo creado: {archivo}")

def configurar_entorno_virtual():
    # Crear entorno virtual
    subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
    
    # Determinar el comando de activación según el SO
    if os.name == 'nt':  # Windows
        pip_path = Path('venv/Scripts/pip.exe')
        activate_cmd = 'venv\\Scripts\\activate'
    else:  # Unix/Linux/Mac
        pip_path = Path('venv/bin/pip')
        activate_cmd = 'source venv/bin/activate'
    
    # Instalar dependencias
    subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], check=True)
    
    print("\nEntorno virtual configurado exitosamente!")
    print(f"Para activar el entorno virtual, ejecuta:")
    print(f"  {activate_cmd}")

if __name__ == "__main__":
    print("Configurando proyecto de visión para palets...")
    crear_estructura()
    configurar_entorno_virtual()
    print("\nProyecto configurado exitosamente!")