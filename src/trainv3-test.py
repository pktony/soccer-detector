from ultralytics import YOLO
import numpy as np
import os

model = YOLO('./train/runs-v3.1/detect/soccer-v3.1/weights/best.pt')

img_path = './test/'


# for i in np.arange(0, 1.05, 0.05):
#     results = model.predict(source=img_path, save=True, conf=i)
#     for r in results:
#         print(f"Detected {len(r)} objects in image")
#         r.save(filename=f"./predict/result-{i}.jpg")

conf = 0.4
results = model.predict(source=img_path, save=False, conf=conf)
i = 0
for r in results:
    os.makedirs(f"./predict/conf_{conf}", exist_ok=True)
    save_path = f"./predict/conf_{conf}/result-{i}.jpg"
    i += 1
    print(f"Detected {len(r)} objects in image -- {os.path.abspath(save_path)}")
    r.save(filename=save_path)