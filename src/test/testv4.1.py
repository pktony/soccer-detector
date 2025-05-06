from model_test_base import test_model


model = test_model(
    model_path='./train/train_results_v4.1/runs/detect/soccer-v4.1/weights/best.pt',
    version_name='v4.1-22',
    # test_path='./test',
    test_path='./dataset/SoccerNet/yolo_dataset/images/valid',
    conf=0.35
)

print(model)