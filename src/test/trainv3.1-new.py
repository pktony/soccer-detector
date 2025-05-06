from model_test_base import test_model


model = test_model(
    model_path='./train/runs-v3.1/detect/soccer-v3.1/weights/best.pt',
    version_name='v3.1-new',
    test_path='./test/',
    conf=0.35
)