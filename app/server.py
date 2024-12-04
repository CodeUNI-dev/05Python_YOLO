import torch
import torch_directml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from app.model import process_frame_with_yolo, load_yolo_model  # Importar la función y el cargador del modelo
import time

# Inicializar FastAPI
app = FastAPI()

# Configurar el dispositivo DirectML (GPU)
device = torch_directml.device()

# Cargar el modelo YOLOv8
model = load_yolo_model()

@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    """
    Maneja la conexión WebSocket para recibir frames, procesarlos con YOLO y enviarlos de vuelta al cliente.
    """
    await websocket.accept()

    try:
        while True:
            # Recibir datos del cliente
            start_time = time.time()  # Inicio de medición de tiempo
            data = await websocket.receive_bytes()
            recv_time = time.time()
            print(f"Tiempo de recepción de datos: {recv_time - start_time:.3f} segundos")

            # Convertir bytes en un frame con OpenCV
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("Error: No se pudo decodificar el frame recibido.")
                continue

            # Procesar el frame con YOLO
            processed_frame, detection_data = process_frame_with_yolo(frame, model)
            if processed_frame is None or not isinstance(processed_frame, np.ndarray):
                print("Error: El modelo no devolvió un frame válido.")
                continue

            # Codificar la imagen procesada a JPEG
            success, encoded_frame = cv2.imencode(".jpg", processed_frame)
            if not success:
                print("Error: No se pudo codificar el frame procesado.")
                continue

            # Enviar el frame codificado al cliente
            send_time = time.time()
            await websocket.send_bytes(encoded_frame.tobytes())
            print(f"Tiempo de envío de datos: {time.time() - send_time:.3f} segundos")
            
    except WebSocketDisconnect:
        print("Cliente desconectado")