# streaming_server.py - Servidor MJPEG para streaming de cámaras en tiempo real
# Permite que el frontend consuma el stream de las cámaras procesadas por la IA

from flask import Flask, Response
import numpy as np
import cv2
import logging
import threading

app = Flask(__name__)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Diccionario para almacenar frames de cada cámara
# Formato: { cam_id: frame_numpy_array }
camera_frames = {}
camera_lock = threading.Lock()

def update_camera_frame(cam_id, frame):
    """
    Actualiza el frame de una cámara específica.
    Debe ser llamado desde detection_service.py después de procesar cada frame.
    
    Args:
        cam_id (str): ID de la cámara
        frame (numpy.ndarray): Frame procesado por OpenCV
    """
    with camera_lock:
        camera_frames[cam_id] = frame.copy()

def remove_camera(cam_id):
    """
    Elimina una cámara del diccionario cuando se detiene el procesamiento.
    
    Args:
        cam_id (str): ID de la cámara a eliminar
    """
    with camera_lock:
        if cam_id in camera_frames:
            del camera_frames[cam_id]
            logger.info(f"📷 Cámara {cam_id} eliminada del streaming")

@app.route('/stream/<cam_id>')
def stream_camera(cam_id):
    """
    Stream MJPEG de una cámara específica.
    El frontend puede consumir este endpoint con una etiqueta <img> o <video>.
    
    Args:
        cam_id (str): ID de la cámara a transmitir
    
    Returns:
        Response: Stream MJPEG multipart
    """
    def generate():
        while True:
            with camera_lock:
                if cam_id in camera_frames:
                    frame = camera_frames[cam_id]
                    # Codificar frame a JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Frame negro con texto si no hay cámara activa
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, f"CAMARA NO DISPONIBLE: {cam_id}", 
                               (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', black_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Pequeña pausa para no saturar CPU
            import time
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health_check():
    """Endpoint para verificar que el servidor está funcionando."""
    with camera_lock:
        cameras_count = len(camera_frames)
    return {
        "status": "ok",
        "active_cameras": cameras_count,
        "camera_ids": list(camera_frames.keys())
    }

if __name__ == '__main__':
    logger.info("🚀 Servidor de streaming MJPEG iniciado en puerto 5001")
    logger.info("📡 Stream disponible en: http://localhost:5001/stream/<cam_id>")
    logger.info("🏥 Health check: http://localhost:5001/health")
    
    app.run(host='0.0.0.0', port=5001, threaded=True)
