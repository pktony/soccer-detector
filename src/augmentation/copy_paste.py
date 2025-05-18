import numpy as np

def copy_paste_ball(image, label, ball_crop, ball_label):
    h, w, _ = image.shape
    x_offset = np.random.randint(0, w - ball_crop.shape[1])
    y_offset = np.random.randint(0, h - ball_crop.shape[0])

    # 붙여넣기
    image[y_offset:y_offset + ball_crop.shape[0], x_offset:x_offset + ball_crop.shape[1]] = ball_crop

    # 새 라벨 좌표 계산
    x_center = (x_offset + ball_crop.shape[1] / 2) / w
    y_center = (y_offset + ball_crop.shape[0] / 2) / h
    w_norm = ball_crop.shape[1] / w
    h_norm = ball_crop.shape[0] / h

    new_label = [ball_label, x_center, y_center, w_norm, h_norm]  # 0 = ball
    label.append(new_label)
    return image, label