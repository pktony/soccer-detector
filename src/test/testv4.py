from model_test_base import test_model


model = test_model(
    model_path='./train/train_results_v4/runs/detect/soccer-v4/weights/best.pt',
    version_name='v4',
    test_path='./test',
    conf=0.3
)

print(model)