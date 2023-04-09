# GoogleNet : Going deeper with convolutions

[블로그에서 보기](https://hyeji0828.github.io/paperreview/GoogleNet_PaperReview/)

## 개요

네트워크의 깊이와 넓이를 증가시키며 연산량은 증가시키지 않는, 멀티 스케일 프로세싱을 하는 Inception 이라는 새로운 구조를 제안한다.

## 1. 소개

최근 3년 간 image recognition과 object detection 등 딥러닝에 많은 발전이 있었지만 이는 새로운 데이터나 더 큰 모델, 발전한 하드웨어 때문이 아니었다. 오히려 깊은 네트워크 구조와 기존의 computer vision 알고리즘의 시너지 덕분이었다.

모바일이나 임베디드 시스템에서의 사용. 이 모델은 실제로 큰 데이터셋에서 현실적으로 사용될 수 있다.

## 2. Related Work

기본적인 CNN 구조는 conv 레이어를 쌓고 (선택적으로 normalization, max-pooling) FC 레이어를 연결하는 것이다. 최근 트렌드는 레이어의 깊이와 사이즈를 증가시키고 드롭아웃을 사용해 오버피팅을 방지하는 것이다.

max-pooling 레이어가 공간적 정보를 잃어버리게 하지만 AlexNet으로 localization, object detection, human pose estimation 등에서 성공적으로 사용되고 있다. 

Robust Obejct Detection 모델과 유사점은 다양한 사이즈의 Gabor filter를 사용해 멀티 스케일 데이터를 처리한다는 점이고, 차이점은 Inception 모델에서는 모든 필터를 학습하며 이 Inception 레이어가 여러 번 반복된다.

네트워크를 쌓는 것은 representation 능력을 향상시키기 위해서다. 이 방법으로 1x1 conv 레이어를 사용하는데 이 모델에서는 두 가지 이유를 위해 사용했다. 1. computational bottleneck을 제거하기 위한 dim 감소를 위해서 2. 깊이뿐 아니라 넓이도 증가

RCNN은 전반적인 Detection 문제를 두 개의 과정으로 함축했다. 1. 색, superpixel consistency 같은low-level 정보를 potential object proposal에 사용하는 것 2. CNN classifier로 object 클래스를 예측하는 것. 위와 같은 방법을 차용하되 두 단계에 추가했다. 1. multi-box prediction과 2. 앙상블 기법으로 더 나은 bbox proposal

- Gabor Filter : 크기와 각도를 인식하는 필터

## 3. Motivation and High Level Considerations

딥 러닝 모델의 성능을 향상 시키는 방법은 깊이와 넓이를 증가시키는 것이다. 하지만 이는 두 가지 문제점이 생기는데 1. 파라미터 수가 크게 증가해 오버피팅이 발생하는 것과 2. 연산 리소스의 증가다. 데이터와 자원이 한정적이기 때문에 아무리 성능 증가가 목적이라도 적당한 정도의 모델 size를 취할 수 밖에 없다.

이 두 가지 문제를 해결하는 근본적인 방법은 Fully Connected를 Sparsely(드물게) Connected 구조로 변경하는 것이다. 데이터 셋의 probability distribution이 크고 sparse한 딥 네트워크로 표현가능할 때 효과적인 네트워크는 / (마지막 레이어의 Activation)과 (결과에 큰 연관을 보이는 뉴런)의 상호관계를 분석하고 / 레이어를 쌓는 것이다. 헤비안 원칙에 의거. 

- 헤비안 원칙 Hebbian Principle : 함께 발화하는 뉴런은 서로 연결된다는 규칙. 신경망 학습에서 어떤 뉴런들이 서로 연결될지를 결정하는 원칙이다. 이를 통해 입출력 사이의 연결성을 최적화한다.

단점은 현재 컴퓨팅 인프라에서는 non-uniform, sparse한 데이터 구조의 계산이 비효율적이라는 것이다. 

## 4. Architectural Details

최적의 local 구조를 찾고 공간적으로 반복하는 것이다. 이전 레이어 unit은 입력 이미지의 어떤 부분 region에 대응된다. 낮은 레이어는 local region에 집중한다. 공간적으로 넓게 퍼진 cluster들이 큰 패치에 의해 커버될 수 있고 큰 region일 수록 패치의 수는 줄어들 것이다. patch-alignment issue를 방지하기 위해 필터사이즈를 1x1, 3x3, 5x5로 제한했다. (편의를 위해서) 이러한 필터에서 생성된 결과들은 concat되어 하나의 output vector를 만들고 다음 stage의 입력값으로 들어간다. 그리고 alternative parallel pooling path를 stage마다 추가해 추가적인 효과를 노렸다.

- path-alignment issue : 패치가 동일한 크기나 위치를 가지지 않아 생기는 문제.
- alternative parallel pooling path : 기존 풀링 레이어가 공간 정보를 압축하는 문제를 보완하기 위해 제안된 풀링 레이어의 한 종류다. 기존 풀링 레이어와 병렬로 동작하면서 디테일한 정보를 보존한다. 이전 레이어의 출력을 다양한 크기로 풀링해 공간적인 정보를 보존하면서도 계산량효율성을 유지한다.

상위 레이어에서는 상위의 abstract한 feature들이 수집된다. 상위 레이어로 가면서 3x3, 5x5의 비율이 증가한다. 

이러한 구조에서는 output filter의 숫자가 이전 스테이지의 필터 수와 일치한다. 풀링 레이어 결과와 conv 레이어 경과를 합치는(merging) 것이 스테이지 마다 output 숫자를 증가시킨다. 비효율적이고 계산량 증가로 이어질 수 있다. 그래서 제안된 구조가 차원 축소와 projection이다. 3x3, 5x5 전에 1x1를 사용함으로써 1.차원을 축소하고 2.ReLU를 사용한다. 

max pooling 이후 1x1은 채널을 맞춰주기 위한 방법

이러한 모듈을 쌓고 중간중간 stride=2의 맥스 풀링을 추가한 것이 전반적인 구조다. 메모리 효율을 위해 Inception 모듈은 상위 레이어에서 사용하고 하위 레이어는 전통적인 cnn 방식을 사용했다.

이러한 구조의 장점은 1. 계산복잡도가 너무 커지지 않게하면서 유닛 수를 증가시키고 2. 차원 축소를 통해 input filter의 수를 지키면서 3. 다양한 크기의 abstract feature를 추출할 수 있도록 다양한 scale을 처리할 수 있다는 것이다.

## 5. GoogLeNet

앙상블 기법이 성능을 꽤 향상시켰다. 정확한 구조적 파라미터의 영향이 적으므로 네트워크의 디테일한 부분은 누락했다.

ReLU를 사용했고 Receptive Field는 224x224 RGB 이미지이다. ‘#3×3 reduce’는 3x3 이전에 1x1이 적용되었다는 의미다. 

학습가능한 파라미터를 가진 Layer는 22개다. classifier 이전에 average pooling을 사용하는 대신 추가적은 Linear layer를 사용했다. FC 레이어를 average pooling 으로 바꾸니 top-1 acc가 0.6 증가했다. 그리고 drop-out은 fc레이어를 제거해도 필수적으로 사용해야한다.

중간 레이어에서 생성된 feature는 잘 구별된다. Auxiliary Classifier를 중간 레이어에 연결해 grdient signal을 강화하고 추가적인 regularization으로 작용했다. 학습 중 auxiliary의 loss는 weight(0.3)을 곱해 total loss에 더해주었다. 추론에서는 auxiliary network는 비활성화된다.

## 6. Training Methodology

DistBelief를 사용해 학습하였다. CPU만 가지고 학습했다. 확률적 경사하강법 (momentum=0.9)로 학습했으며 추론 단계에서는 최종 모델을 만들기 위해 Polyak averaging을 사용했다. 학습률은 8epoch마다 4%씩 감소시켰다.

- DistBelief : Tensorflow의 전 버전.
- Polyak averaging : 가중치를 평균화하는 방법. 모델의 안정성과 일반화 성능을 향상시킨다.

학습 방법은 모델에 따라서 바뀌어서 정확히 어떤 방식에 영향을 받았다고 말하기 힘들다. 사용한 학습 방법은 아래와 같다.

1. 8~100%에 균등하게 분포된 사이즈의 다양한 이미지 패치
2. 이미지 비율은 3/4~4/3 사이 랜덤
3. photometric distortion
4. 무작위 interpolation 방법 (bilinear, area, nearest neighbor, cubic)으로 resize
- photometric distortion : 광도(phtometric) 왜곡. 밝기, 색조, 채도, 대비, 노이즈 등을 조정해 새로운 이미지를 만드는 방식
- image interpolation : 이미지 보간. 픽셀값을 바꾸거나 사이즈를 변경하는 방식.

## 7. ILSVRC 2014 Classification Challenge Setpup and Results

1000개의 하위 카테고리를 분류하는 classification 대회. top-5 에러를 기준으로 판정함.

학습 데이터=1.2M / validation 데이터=5만 / 테스트 데이터=10만

대회에 참가하며 새 데이터를 사용하지는 않았지만 성능을 향상시키기 위해서 여러 학습 테크닉을 적용했다.

1. 7개 (한 개의 wider한 모델 포함)의 GoogleNet으로 앙상블 기법 사용. 초기값은 모두 같게 학습했으나 샘플링 방식이나 입력 이미지의 순서는 달랐다.
2. 이미지를 4개의 사이즈 [256,288,320,352]로 조정하고 [left, center, right]로 3개의 square를 만든다. (사람 이미지의 경우 [top, center, bottom]) Square 마다 4개의 코너 + 중앙 224x224 Crop + Square를 224x224로 resize한 것. 총 6개. 이 이미지를 mirror한 것 까지 포함. = 4x3x6x2=144개
3. Crop에 대한 softmax를 평균내고, 모델별로 평균을 내었다. 데이터 crop을 max pooling 하거나 전체 모델에 대해 데이터를 평균내는 방식은 그냥 평균을 내는 것보다 못한 결과였다.

## 8. ILSVRC 2014 Detection Challenge Setup and Results

bounding box를 생성하고 200개의 클래스에서 맞추는 방식. 클래스를 맞추고 ground truth의 bbox와 50%이상 겹치면 정답이라고 판정했다. 관계없는 detection은 false positive로 간주하고 페널티를 주었다.

- Jaccard index : 두 집합의 유사도를 판정하는 방식. 두 집합의 합집합에서 교집합이 어느정도의 비율을 차지하는지 나타낸다.

이미지에는 오브젝트가 많거나 아애 없기도 하고 오브젝트의 크기가 다양했다. 결과는 mAP로 측정되었다.

R-CNN과 유사하지만 Inception 모델을 region classifier로 사용했다. region proposal 단계에서 selective search 방식과 함께 multi-box prediction 방식을 사용했다. 

- multi-box prediction : Yolo, SSD(Single Shot multibox Detector)에서 사용하는 방식으로 격자를 나누고 격자마다 bbox를 예측하는 방식.

False positive 수를 줄이기 위해서 superpixel 사이즈를 2배 증가시켰다. → selective search 결과를 반으로 줄이는 효과가 있었다. multi-box 결과와 더했다.

- SuperPixel : 인접한 비슷한 픽셀들을 묶어 그룹화 한 것.

결과적으로 R-CNN보다 60%의 proposal을 사용하면서도 92→93%로 범위를 증가시켰다. 

proposal 수를 줄이고 범위를 확장시켜 1%의 향상을 가져왔다. R-CCN과는 다르게 bbox regression을 사용하지 않았다. (시간 부족으로)

## 9. Conclusion

Sparse Structure와 dense block으로 성능을 향상시킬 수 있었다. 적당한 연산량 증가로 더 얕거나 좁은 네트워크에 비해 더 좋은 성능을 얻었다. context나 bbox regression을 사용하지 않았음에도 더 좋은 성능을 보였다.
