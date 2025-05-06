from model_test_base import test_model


model = test_model(
    model_path='./train/train_results_v3.1-new/runs/detect/soccer-v3.1-new/weights/best.pt',
    version_name='v3.1-new',
    test_path='./dataset/test',
    conf=0.3
)

print(model.summary())