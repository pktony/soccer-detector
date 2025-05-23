from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt', verbose=True)
# results = model("https://ultralytics.com/images/bus.jpg", save=True)

# results[0].save(filename="result.jpg")


# try :
#     model.train(
#         data="/Users/psangwon/Documents/GitRepository/python/soccer-detector/dataset/data.yaml",
#         epochs=100,
#         batch=16,
#         imgsz=640
#     )
# except Exception as e :
#     print(e)


from util.video_util import read_video, save_video
from tracker import Tracker

video_frames = read_video('./08fd33_4.mp4')

tracker = Tracker(model_path='./train/train_results_v6/train6/weights/best.pt')
tracks = tracker.get_object_tracks(video_frames, read_from_stub = True, stub_path = 'stubs/cam_move.pkl')

for track_id, player in tracks['player'][0].items():
    bbox = player['bbox']
    frame = video_frames[0]

    cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # [bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cv2.imwrite(f'.cropped_image.jpg', cropped_image)
    break

output_video_frames = tracker.draw_annotations(video_frames, tracks)

save_video(output_video_frames, './result-11.mp4')
