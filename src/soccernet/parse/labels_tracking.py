import os
import shutil
from collections import OrderedDict
from visualize.visualize_labels import draw_yolo_boxes_on_image


class_convert_map = {
    "ball": [],
    "goalkeeper": [],
    "player": [],
    "referee": []
}

yolo_class_map = OrderedDict({
    "ball": 0,
    "goalkeeper": 1,
    "player": 2,
    "referee": 3
})

def get_class_convert(cls_index):
    for k, v in class_convert_map.items():
        if cls_index in v:
            return yolo_class_map[k]

def convert_gt_to_yolo(dataset_dir, output_dir):
    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    viz_out = os.path.join(output_dir, "viz")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    os.makedirs(viz_out, exist_ok=True)

    # 클래스 이름을 추출하고 고유하게 저장 (순서 유지)
    reset_class_info()
    
    class_name_set = OrderedDict()

    for seq in os.listdir(dataset_dir):
        seq_path = os.path.join(dataset_dir, seq)
        if not os.path.isdir(seq_path):
            continue

        gameinfo_path = os.path.join(seq_path, "gameinfo.ini")
        gt_path = os.path.join(seq_path, "gt/gt.txt")
        seqinfo_path = os.path.join(seq_path, "seqinfo.ini")
        img_dir = os.path.join(seq_path, "img1")

        if not os.path.exists(gt_path) or not os.path.exists(img_dir):
            continue

        # 🔹 해상도 설정
        width, height = calculate_img_size(seqinfo_path)

        # gameinfo.ini → 클래스 이름 추출
        class_name_set = OrderedDict((k, None) for k in yolo_class_map.keys())
        process_class_info(gameinfo_path)

        # 라벨 처리
        frame_labels = dict()
        process_label(seq, gt_path, width, height, frame_labels)

        # 🔹 라벨 파일 저장
        save_label_files(labels_out, frame_labels)

        for label_name, lines in frame_labels.items():
            frame_id = label_name.split("_")[-1].split(".")[0]
            img_src = os.path.join(img_dir, f"{frame_id}.jpg")
            img_dst = os.path.join(images_out, f"{label_name.replace('.txt', '.jpg')}")
            if os.path.exists(img_src):
                shutil.copyfile(img_src, img_dst)

        draw_yolo_boxes_on_image(
            image_path=img_src,
            label_lines=lines,
            output_path=os.path.join(viz_out, os.path.basename(img_dst)),
            class_name_list=list(class_name_set.keys())
        )

        print(f"✅ Processed {seq}")

    # 🔹 class name list 출력
    class_names = list(class_name_set.keys())
    print("\n✅ YOLO class names:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    return class_names

def calculate_img_size(seqinfo_path):
    width, height = 1920, 1080
    if os.path.exists(seqinfo_path):
        with open(seqinfo_path, "r") as f:
            for line in f:
                if line.startswith("imWidth"):
                    width = int(line.split("=")[1])
                elif line.startswith("imHeight"):
                    height = int(line.split("=")[1])
    return width,height

def save_label_files(labels_out, frame_labels):
    for label_name, lines in frame_labels.items():
        frame_id = label_name.split("_")[-1].split(".")[0]
        label_path = os.path.join(labels_out, label_name)
        with open(label_path, "w") as out_f:
            out_f.write("\n".join(lines))

def process_label(seq, gt_path, width, height, frame_labels):
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue

            frame, cls, x, y, w, h, conf, x3d, y3d, z3d = parts[:10]
            frame = int(frame)
            cls = get_class_convert(int(cls))
            x, y, w, h = map(float, (x, y, w, h))

                # YOLO format: class_id, x_center, y_center, width, height
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            label_name = f"{seq}_{frame:06d}.txt"
            yolo_line = f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            frame_labels.setdefault(label_name, []).append(yolo_line)

def process_class_info(gameinfo_path):
    if os.path.exists(gameinfo_path) == False:
        return
    
    with open(gameinfo_path, "r") as f:
        for line in f:
            if line.startswith("trackletID") == False:
                continue
            
            tracklet_id = int(line.split('=')[0].split('_')[1])
            cls_name = line.split('=')[1]

            if ('ball' in cls_name):
                class_convert_map['ball'].append(tracklet_id)
            elif ('player' in cls_name):
                class_convert_map['player'].append(tracklet_id)
            elif ('referee' in cls_name):
                class_convert_map['referee'].append(tracklet_id)
            elif ('goalkeeper' in cls_name):
                class_convert_map['goalkeeper'].append(tracklet_id)

def reset_class_info():
    class_convert_map['ball'] = []
    class_convert_map['player'] = []
    class_convert_map['referee'] = []
    class_convert_map['goalkeeper'] = []

# # 사용 예시
# convert_gt_and_copy_images(
#     dataset_dir="./dataset/SoccerNet_Tracking",
#     output_dir="./dataset/SoccerNet_Tracking_yolo"
# )
