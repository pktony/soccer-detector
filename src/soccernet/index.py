from unzip import unzip_soccerNet
from parse_labels import parse_labels_recursive
from downloader import download_soccerNet


download_soccerNet('./dataset/SoccerNet')
unzip_soccerNet('./dataset/SoccerNet')
parse_labels_recursive(
    local_path='./dataset/SoccerNet',
    output_path = './dataset/SoccerNet/yolo_dataset')