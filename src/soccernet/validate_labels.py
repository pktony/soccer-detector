def validate_labels():
    for line in f:
        cls, x, y, w, h = map(float, line.strip().split())
        assert 0 <= x <= 1 and 0 <= y <= 1
        assert 0 < w <= 1 and 0 < h <= 1
