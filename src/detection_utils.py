# src/detection_utils.py
import cv2
import yaml
import os

def load_camera_config():
    """Carga la configuración de la cámara con manejo de errores"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_settings.yaml')
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception(f"Archivo de configuración no encontrado: {config_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error en formato YAML: {e}")

def enhanced_edge_detection(image):
    """Detección de bordes mejorada para mayor precisión"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    return cv2.Canny(blurred, 50, 150)