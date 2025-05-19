import cv2
import numpy as np
import os
import yaml
from detection_utils import load_camera_config

class Calibrator:
    def __init__(self):
        self.cam_config = load_camera_config()
        self.points = []
        self.homography_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'calibration', 
            'homography.npy'
        )
        
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Punto {len(self.points)} seleccionado: ({x}, {y})")
    
    def calculate_homography(self):
        src = np.array(self.points, dtype=np.float32)
        dst = np.array([
            [0, 0],            # Superior izquierda
            [1000, 0],         # Superior derecha
            [1000, 1000],      # Inferior derecha
            [0, 1000]          # Inferior izquierda
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(src, dst)
        np.save(self.homography_path, H)
        print(f"Matriz de homografía guardada en: {self.homography_path}")
        return H
    
    def run_calibration(self):
        cap = cv2.VideoCapture(self.cam_config['stream_url'])
        if not cap.isOpened():
            print("Error: No se puede conectar a la cámara")
            return
        
        cv2.namedWindow("Calibración - Seleccione 4 puntos")
        cv2.setMouseCallback("Calibración - Seleccione 4 puntos", self.click_event)
        
        print("Instrucciones:")
        print("1. Seleccione 4 puntos en este orden:")
        print("   - Esquina superior izquierda del área de trabajo")
        print("   - Esquina superior derecha")
        print("   - Esquina inferior derecha")
        print("   - Esquina inferior izquierda")
        print("2. Presione ESC para finalizar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo obtener frame")
                break
            
            # Dibujar puntos seleccionados
            for i, (x, y) in enumerate(self.points):
                color = (0, 255, 0) if i < 3 else (0, 0, 255)
                cv2.circle(frame, (x, y), 10, color, -1)
                cv2.putText(frame, str(i+1), (x + 20, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Calibración - Seleccione 4 puntos", frame)
            
            key = cv2.waitKey(1)
            if key == 27 or len(self.points) == 4:  # Tecla ESC o 4 puntos seleccionados
                break
        
        if len(self.points) == 4:
            self.calculate_homography()
        else:
            print("Calibración cancelada o incompleta")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = Calibrator()
    calibrator.run_calibration()