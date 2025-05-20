from ultralytics import YOLO    
import supervision as sv
import cv2

from util import get_center_bbox, get_bbox_width

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
    

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        x_center, y_center = get_center_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(width, 0.35 * width),
            angle=0,
            startAngle=45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
            )
        
        return frame

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):
        detections = self.detect_frames(frames)

        tracks = {
            "ball": [],
            "player": [],
            "referee": [],
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # (0, ball) (1, goalkeeper)...
            cls_names_inv = {v:k for k, v in cls_names.items()} # (ball:0, goalkeeper:1...)

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for class_id in enumerate(detection_supervision.class_id):
                print(class_id)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['player'].append({})
            tracks["ball"].append({})
            tracks['referee'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player'] :
                    tracks['player'][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referee'][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox": bbox}


            # print(detection_supervision)

        return tracks
    

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['player'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referee'][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)

            output_video_frames.append(frame)

        return output_video_frames
