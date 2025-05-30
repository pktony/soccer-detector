from ultralytics import YOLO
from ultralytics.engine.model import Model
import os

def test_model(model_path, version_name, test_path, conf) -> Model:
    model = YOLO(model=model_path, verbose=True)

    results = model.predict(source=test_path, save=False, conf=conf)
    i = 0
    for r in results:
        os.makedirs(f"./predict/{version_name}", exist_ok=True)
        save_path = f"./predict/{version_name}/result-{conf}-{i}.jpg"
        i += 1
        print(f"Detected {len(r)} objects in image -- {os.path.abspath(save_path)}")
        r.plot(font_size=1, line_width=1,pil=True)
        r.save(filename=save_path)

    return model