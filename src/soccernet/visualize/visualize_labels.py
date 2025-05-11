from PIL import Image, ImageDraw, ImageFont

def draw_yolo_boxes_on_image(image_path, label_lines, output_path, class_name_list):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for line in label_lines:
        cls_id, xc, yc, w, h = map(float, line.strip().split())
        xc *= img_w
        yc *= img_h
        w *= img_w
        h *= img_h
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        label = class_name_list[int(cls_id)] if int(cls_id) < len(class_name_list) else str(int(cls_id))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="yellow", font=font)

    image.save(output_path)