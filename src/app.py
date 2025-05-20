from ultralytics import YOLO

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
tracks = tracker.get_object_tracks(video_frames)

output_video_frames = tracker.draw_annotations(video_frames, tracks)

save_video(video_frames, './result.mp4')
