# Sistema principal de visi�n por computadora\n
import cv2
import numpy as np
from detection_utils import load_camera_config, detect_pallets
from kuka_communicator import KUKAConnector

class VisionSystem:
    def __init__(self):
        self.cam_config = load_camera_config()
        self.kuka = KUKAConnector()
        self.H = np.load("../data/calibration/homography.npy")
    
    def pixel_to_world(self, x, y):
        point = np.array([[[x, y]]], dtype=np.float32)
        return cv2.perspectiveTransform(point, self.H)[0][0]
    
    def run(self):
        cap = cv2.VideoCapture(self.cam_config["stream_url"])
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Detección de palets
            contours = detect_pallets(frame)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 5000:
                    M = cv2.moments(cnt)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    
                    # Convertir a coordenadas reales
                    wx, wy = self.pixel_to_world(cx, cy)
                    
                    # Enviar a KUKA
                    self.kuka.send_position(wx, wy)
                    
                    # Dibujar resultados
                    cv2.circle(frame, (cx, cy), 10, (0,255,0), -1)
                    cv2.putText(frame, f"{wx:.0f},{wy:.0f}mm", 
                               (cx-100, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0,255,0), 2)
            
            cv2.imshow("Sistema de Visión", frame)
            if cv2.waitKey(1) == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vs = VisionSystem()
    vs.run()