# from downloader import download_tracking
from unzip import unzip_soccerNet
from parse.labels_tracking import convert_gt_to_yolo

try :
    # download_tracking('./dataset/SoccerNet_Tracking')
    unzip_soccerNet('./dataset/SoccerNet_Tracking')
    convert_gt_to_yolo(
        dataset_dir='./dataset/SoccerNet_Tracking',
        output_dir = './dataset/SoccerNet_Tracking/yolo_dataset')
except Exception as e :
    print(e)