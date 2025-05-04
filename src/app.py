from ultralytics import YOLO

model = YOLO('yolov8n.pt', verbose=True)
# results = model("https://ultralytics.com/images/bus.jpg", save=True)

# results[0].save(filename="result.jpg")

try :
    model.train(
        data="/Users/psangwon/Documents/GitRepository/python/soccer-detector/dataset/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640
    )
except Exception as e :
    print(e)