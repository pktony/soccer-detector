# Soccer Detector - YOLOv8 모델 학습 기록

## Class 목록

* `player`
* `goalkeeper`
* `ball`
* `referee`

---

## Version History

---

### v1

* 혼자 수집 및 라벨링한 50개의 이미지로 최초 학습
* 최초 학습 모델 → 이후 모든 버전의 베이스가 된 모델

#### Hyperparameter

```yaml
epochs: 150
batch: 16
imgsz: 640
patience: 30
```

#### Result

* metrics/mAP50: **0.47**
* metrics/mAP50-95: **0.28**
* 클래스 구분 성능이 낮음 ( 데이터 수 부족 )
* "referee"와 "goalkeeper"의 혼동 다수 발생

---

### v2

* `v1`을 **fine-tuning**
* 추가 데이터 출처: [Roboflow Football Dataset v3](https://universe.roboflow.com/general-1zdku/football-player-detection_v3)

#### Hyperparameter

```yaml
epochs: 150
batch: 16
imgsz: 640
patience: 30
```

#### Result

* metrics/mAP50: **0.63**
* metrics/mAP50-95: **0.41**
* 전체적으로 성능 향상
* "player"와 "ball" 탐지 정확도 상승
* 여전히 배경 노이즈에 민감함

---

### v3

* `v1`을 **fine-tuning**
* 동일한 Roboflow 데이터셋 사용
* **유니폼 색상 차이의 영향을 줄이기 위해 HSV 증강 사용**

#### Hyperparameter

```yaml
epochs: 150
batch: 16
imgsz: 640
patience: 30

hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
```

#### Result

* metrics/mAP50: **0.67**
* metrics/mAP50-95: **0.44**
* 다양한 유니폼 색상에도 robustness 증가
* "goalkeeper"와 "referee" 구분이 좀 더 안정적임
* mixup이나 mosaic 없이도 데이터 다양성 확보

---

### v3.1

* `v1`을 **fine-tuning**
* 동일한 Roboflow 데이터셋 사용
* 기존보다 좀 더 긴 학습 (epoch 200), early stopping 기준 완화

#### Hyperparameter

```yaml
epochs: 200
batch: 16
imgsz: 640
patience: 50
```

#### Result

* metrics/mAP50: **0.69**
* metrics/mAP50-95: **0.46**
* overfitting 경향을 줄이기 위해 early stopping 여유 설정
* 검증 성능 안정화됨
* 다소 느리지만 안정적인 수렴 확인

---

### v3.1 - new

* 기존 pretrained 모델 없이 **YOLOv8n을 처음부터 학습**
* 동일한 Roboflow 데이터셋 사용

#### Hyperparameter

```yaml
epochs: 200
batch: 16
imgsz: 640
patience: 50
```

#### Result

* metrics/mAP50: **0.64**
* metrics/mAP50-95: **0.39**
* pretrained 없이도 reasonable한 성능 확보
* fine-tuning 버전보다 수렴 속도 느림
* 초반 학습 성능 낮았으나, 후반 성능 점차 향상

---

## 참고

* [Ultranalytics YOLO v8](https://docs.ultralytics.com/ko/models/yolov8/)
