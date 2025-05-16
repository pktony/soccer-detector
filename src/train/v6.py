from ultralytics import YOLO

model = YOLO('yolov8n.pt', verbose=True)

try :
    results = model.train(
        data="./dataset/data.yaml",
        epochs=200,
        batch=512,
        lr0=0.0001,
        imgsz=640,
        patience=50,
    )

    print(results)
except Exception as e :
    print(e)