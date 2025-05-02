from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model("https://ultralytics.com/images/bus.jpg", save=True)

results[0].save(filename="result.jpg")