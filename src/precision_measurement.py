import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from detection_utils import load_camera_config

class PrecisionAnalyzer:
    def __init__(self):
        self.cam_config = load_camera_config()
        
        # Cargar parámetros de calibración
        calib_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration', 'homography.npy')
        if not os.path.exists(calib_path):
            raise FileNotFoundError("Primero ejecuta calibration_tool.py")
            
        self.H = np.load(calib_path)
        
    def measure_object(self, img_path, true_width_mm):
        # Verificar existencia de imagen
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")
            
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Error leyendo la imagen")
        
        # Procesamiento mejorado
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No se detectaron contornos")
            
        main_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)
        
        # Transformación de coordenadas
        world_points = cv2.perspectiveTransform(np.array([box], dtype=np.float32), self.H)
        
        width = np.linalg.norm(world_points[0][0] - world_points[0][1])
        height = np.linalg.norm(world_points[0][1] - world_points[0][2])
        
        self.plot_results(img, box, width, height, true_width_mm)
        return width, height

    def plot_results(self, img, box, width, height, true_width):
        plt.figure(figsize=(15, 7))
        
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Imagen Original")
        
        plt.subplot(122)
        display_img = img.copy()
        cv2.drawContours(display_img, [np.int0(box)], 0, (0,255,0), 3)
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Ancho: {width:.2f}mm | Alto: {height:.2f}mm\nReal: {true_width}mm | Error: {abs(width-true_width):.2f}mm")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyzer = PrecisionAnalyzer()
    
    # Usar imagen de prueba (ajusta la ruta)
    image_path = os.path.join("data", "images", "calib_object.jpg")
    true_width = 200  # Valor real en mm
    
    try:
        w, h = analyzer.measure_object(image_path, true_width)
        print(f"""
        Resultados:
        - Medido: {w:.2f}mm x {h:.2f}mm
        - Real: {true_width}mm
        - Error absoluto: {abs(w - true_width):.2f}mm
        """)
    except Exception as e:
        print(f"Error: {str(e)}")
