import socket
import yaml

class KUKAConnector:
    def __init__(self):
        self.load_config()
        self.connect()
        
    def load_config(self):
        with open("../config/kuka_config.yaml") as f:
            self.config = yaml.safe_load(f)
        
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.config["ip"], self.config["port"]))
        print("Conectado a KUKA")
    
    def send_position(self, x, y):
        command = f"{{X {x}, Y {y}, Z {self.config['default_z']}}"
        self.sock.sendall(command.encode())
    
    def __del__(self):
        self.sock.close()