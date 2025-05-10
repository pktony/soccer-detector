from downloader import download_tracking
from unzip import unzip_soccerNet
from parse_labels import parse_labels_recursive

try :
    download_tracking('./dataset/SoccerNet_Tracking')
    unzip_soccerNet('./dataset/SoccerNet_Tracking')
    parse_labels_recursive(
        local_path='./dataset/SoccerNet_Tracking',
        output_path = './dataset/SoccerNet_Tracking/yolo_dataset')
except Exception as e :
    print(e)