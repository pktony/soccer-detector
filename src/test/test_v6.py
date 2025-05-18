

from ultralytics import YOLO


model = YOLO('./runs/detect/train5/weights/best.pt', verbose=True)

results = model.predict(
    './08fd33_4.mp4',
    save=True)