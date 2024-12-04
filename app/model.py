import torch
import cv2
import numpy as np

def load_yolo_model():
    # Cargar el modelo YOLOv8 aquí
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)  # Ejemplo de carga del modelo
    print('MODEL',model)
    return model

def process_frame_with_yolo(frame, model):
    try:
        # Procesar el frame con el modelo YOLOv8
        results = model(frame)

        detection_data = []
        for result in results.xyxy[0]:  # Acceder a los resultados correctamente
            # Coordenadas y dimensiones de la detección
            xyxy = result[:4].cpu().numpy()
            conf = result[4].cpu().item()
            cls = int(result[5].cpu().item())

            # Coordenadas para dibujar en el frame
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{model.names[cls]} ({conf * 100:.1f}%)"

            # Dibuja la caja en el frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Agregar datos de la detección
            detection_data.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": cls,
                "label": model.names[cls]
            })

        return frame, detection_data

    except Exception as e:
        print(f"Error procesando el frame con YOLOv8: {e}")
        return frame, []