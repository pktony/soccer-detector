from ultralytics import YOLO    
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()


    def detect_frames(self, frames):
        # memory 이슈로 batch를 나눠서 진행
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1, verbose=False)
            detections += detections_batch
            break
        
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # (0, ball) (1, goalkeeper)...
            cls_names_inv = {v:k for k, v in cls_names.items()} # (ball:0, goalkeeper:1...)

            detection_supervision = sv.Detections.from_ultralytics(detection)

            print(detection_supervision)
