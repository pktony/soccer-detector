import os
import json
import shutil

# 클래스 매핑
CLASS_MAP = {
    "Ball": 0,
    "Goalkeeper team left": 1,
    "Goalkeeper team right": 1,
    "Player team left": 2,
    "Player team right": 2,
    "Main referee": 3,
    "Side referee": 3,
}

#names: ['ball', 'goalkeeper', 'player', 'referee']

SETS = ["train", "valid", "test"]

def create_directory(output_path):
    for s in SETS:
        os.makedirs(os.path.join(output_path, "images", s), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels", s), exist_ok=True)

def find_image_file(base_dir, filename):
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def parse_single_label_file(label_json_path, section_data, output_path):
    # 폴더 이름 추출: ~/framev3/001/Labels-v3.json → prefix = "001"
    dir_name = os.path.dirname(label_json_path)
    prefix = os.path.basename(dir_name)

    for img_name, info in section_data.items():
        image_set = info["imageMetadata"]["set"]
        if image_set not in SETS:
            continue

        width = info["imageMetadata"]["width"]
        height = info["imageMetadata"]["height"]

        label_lines = []
        player_count = 0
        referee_count = 0
        ball_count = 0
        for box in info.get("bboxes", []):
            cls_name = box["class"]
            if cls_name not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cls_name]

            if cls_id == 1 or cls_id == 2:
                player_count += 1
            if cls_id == 3:
                referee_count += 1
            if cls_id == 0:
                ball_count += 1
            

            pts = box["points"]
            x1, x2 = pts["x1"], pts["x2"]
            y1, y2 = pts["y1"], pts["y2"]

            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            if not (
                0 <= x_center <= 1 and
                0 <= y_center <= 1 and
                0 < w <= 1 and
                0 < h <= 1
                ):
                print(f'[!] 유효한 라벨을 찾을 수 없음 - ({x_center} {y_center})        ({w} {h})')
                continue

            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # 잘못된 라벨일 경우: 건너뜀 (파일 저장 & 이미지 복사 X)
        if len(label_lines) == 0 or player_count > 22 or referee_count > 3 or ball_count > 1:
            print(f"[!] {prefix}_{img_name} → {len(label_lines)} player: {player_count} referree: {referee_count} ball:{ball_count}")
            continue

        # 파일 이름 변경 (충돌 방지)
        unique_img_name = f"{prefix}_{img_name}"
        label_file_path = os.path.join(output_path, "labels", image_set, unique_img_name.replace(".png", ".txt"))
        with open(label_file_path, "w") as f:
            f.write("\n".join(label_lines))

        # 이미지 복사
        found_img_path = find_image_file(dir_name, img_name)
        if found_img_path:
            dst_img = os.path.join(output_path, "images", image_set, unique_img_name)
            shutil.copy(found_img_path, dst_img)
        else:
            print(f"[!] 이미지 {img_name} 찾을 수 없음")

def parse_labels_recursive(local_path, output_path):
    create_directory(output_path)
    for root, _, files in os.walk(local_path):
        if "Labels-v3.json" in files:
            label_json_path = os.path.join(root, "Labels-v3.json")

            with open(label_json_path, "r") as f:
                data = json.load(f)
            
            print(f"[+] 처리 중: {label_json_path}")
            parse_single_label_file(label_json_path, data['actions'], output_path)
            parse_single_label_file(label_json_path, data['replays'], output_path)

# 실행 예시
if __name__ == "__main__":
    LOCAL_PATH = "dataset/SoccerNet"   # unzip된 최상위 디렉토리
    OUTPUT_PATH = "yolo_dataset"

    parse_labels_recursive(LOCAL_PATH, OUTPUT_PATH)
