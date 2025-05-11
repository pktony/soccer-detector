from soccernet.unzip import unzip_soccerNet
from soccernet.parse_labels_spotting import parse_labels_recursive
from downloader import download_action_spotting


download_action_spotting('./dataset/SoccerNet')
unzip_soccerNet('./dataset/SoccerNet')
parse_labels_recursive(
    local_path='./dataset/SoccerNet',
    output_path = './dataset/SoccerNet/yolo_dataset')