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

### v4

* 동일한 Roboflow 데이터셋 사용
  * 학습 / 검증 데이터 비중을 변경하고, 재 학습 (train ⬆️)
* Pre-processing
  * 640x640 stretch
  * auto orientation : 사진의 메타 데이터와 실제 방향이 잘못된 경우 보정을 위해


#### Result
```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    121/200      3.59G     0.9012     0.4655     0.8437         34        640: 100%|██████████| 71/71 [00:22<00:00,  3.12it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 9/9 [00:02<00:00,  3.20it/s]                   all        258       1694      0.892       0.89      0.921       0.62
EarlyStopping: Training stopped early as no improvement observed in last 50 epochs. Best results observed at epoch 71, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

Model summary (fused): 72 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 9/9 [00:05<00:00,  1.67it/s]
                   all        258       1694      0.924      0.849      0.927      0.629
                  ball        253        523      0.937      0.911      0.938      0.719
            goalkeeper         37         37      0.884      0.821      0.926      0.589
                player         51       1012      0.962      0.945       0.98      0.692
               referee         51        122      0.912      0.721      0.866      0.519
```

* player와 ball은 거의 완벽에 가까운 성능을 보여줌.
* goalkeeper도 좋은 성능이지만, 표본 수(37개)가 적어서 일반화에 주의.
* referee는 상대적으로 재현율과 mAP가 낮은데, 아마 이미지 내 등장 비율이 적거나 위치가 다양해서 그럴 수 있음.

## v4.1

* SoccerNet에서 가져온 데이터로 fine tuning

#### Hyper Parameter
```yaml
epochs: 200
batch: 16
imgsz: 640
patience: 50
```

#### Result

* 전체적으로 성능이 줄었지만, 몇 개 샘플로 테스트해보면 더 잘 검출함
  * precision은 오히려 떨어지면서, background를 오탐하는 경우가 늘어남
  * 아예 검출을 못하던 물체까지 탐지할 수 있게 됨
* 원본 이미지 파일을 640으로 압축하다보니 작은 물체(공), 초점이 흔들린 물체 

```
Model summary (fused): 72 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:10<00:00,  1.19it/s]
                   all        377       4030      0.734      0.657      0.691      0.471
                  ball        285        285      0.825      0.368      0.473      0.271
            goalkeeper        256        265      0.468      0.592      0.526      0.364
                player        372       3244      0.863      0.894      0.928      0.665
               referee        179        236      0.778      0.773      0.835      0.583
```

| 지표                | fine tune (`v4`) | SoccerNet fine-tune (`v4.1`) | 변화 추이   |
| ----------------- | ----------- | ---------------- | ------- |
| **mAP50 (전체)**    | 0.927       | 0.691            | ⬇ 크게 하락 |
| **mAP50-95**      | 0.629       | 0.471            | ⬇ 하락    |
| **Box Precision** | 0.924       | 0.734            | ⬇ 하락    |
| **Recall**        | 0.849       | 0.657            | ⬇ 하락    |


## 참고
* [SoccerNet] (https://www.soccer-net.org/tasks/tracking)
* [Ultranalytics YOLO v8](https://docs.ultralytics.com/ko/models/yolov8/)
