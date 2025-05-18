import cv2


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
    return frames


def save_video(output_video_frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps=24,
        frameSize=(output_video_frames[0].shape[1], output_video_frames[0].shape[0])
        )
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()