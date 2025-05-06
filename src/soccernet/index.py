from unzip import unzip_soccerNet
from parse_labels import parse_labels_recursive


unzip_soccerNet('./dataset/SoccerNet')
parse_labels_recursive(
    local_path='./dataset/SoccerNet',
    output_path = './dataset/SoccerNet/yolo_dataset')