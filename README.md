# PokéVision: 150종 포켓몬 분류기 (Pokemon Classifier)

## 프로젝트 소개 (Project Overview)
주어진 포켓몬 이미지의 픽셀 특징을 분석하여 150종의 포켓몬 중 어떤 포켓몬인지 정확한 이름을 예측하는 컴퓨터 비전 분류 애플리케이션입니다.

딥러닝 프레임워크(PyTorch)와 사전 학습된(Pre-trained) CNN Backbone 모델을 활용한 **전이 학습(Transfer Learning)** 기술을 적용했습니다. 이를 통해 처음부터 모델을 학습(From Scratch)하는 것보다 훨씬 적은 컴퓨팅 자원과 시간으로 높은 분류 정확도를 달성했습니다.

* **데이터셋 (Dataset):** Kaggle 7,000 Labeled Pokemon (Total 150 Classes)
* **개발 환경:** Python 3.13, PyTorch, Streamlit, macOS (Apple Silicon M1 Pro / MPS Acceleration 지원)

---

## 실험 설정 및 성능 비교 (Experimental Setups)
본 프로젝트는 모델의 아키텍처와 전이 학습(Fine-tuning) 범위를 변수로 설정하여 총 4가지의 독립적인 실험을 진행했습니다.

1. **[실험 A] ResNet18 (Feature Extraction):** ImageNet 데이터로 사전 학습된 가중치를 동결(Freeze)하고, 마지막 분류기(FC layer)만 150개 클래스에 맞게 학습했습니다.
2. **[실험 B] ResNet18 (Fine-Tuning):** 사전 학습된 가중치를 초기값으로 사용하되, 전체 네트워크의 가중치를 포켓몬 데이터에 맞게 미세 조정(Fine-tuning)했습니다.
3. **[실험 C] MobileNet V3 Small (Fine-Tuning):** ResNet 대비 파라미터 수가 약 1/7로 매우 적은 경량화 모델을 사용하여 파인튜닝을 진행했습니다.
4. **[실험 D] ResNet18 (From Scratch):** 사전 학습 가중치를 전혀 사용하지 않고 무작위 초기화 상태에서 학습하여, 전이 학습의 효과를 대조군으로서 검증했습니다.

### 최종 성능 평가 표
> **Note:** 테스트셋을 분리하여 Precision, Recall, F1-Score를 측정했습니다.

| 실험 | 설정 (Backbone & 방식) | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| A | ResNet18 (Feature Extraction) | 0.7522 | 0.7851 | 0.7630 | 0.7484 |
| B | ResNet18 (Fine-Tuning) | 0.8314 | 0.8695 | 0.8367 | 0.8315 |
| C | MobileNet V3 (Fine-Tuning) | 0.8695 | 0.8912 | 0.8698 | 0.8634 |
| D | ResNet18 (From Scratch) | 0.2302 | 0.2474 | 0.2366 | 0.1887 |


---

## 데모 GUI (Streamlit Application)
사용자가 직접 포켓몬 이미지를 업로드하고 결과를 확인할 수 있는 인터랙티브 웹 데모를 구현했습니다. 
가장 성능이 높았던 모델(`best_pokemon_model.pt`)의 아키텍처를 자동으로 인식하여 가중치를 로드하며, 상위 5개(Top-5)의 예측 확률을 제공합니다.

### 스크린샷 및 시연 영상
![Demo Screenshot](<여기에_이미지_경로_또는_URL_삽입>)


---

## 실행 방법 

**1. 패키지 설치**
```bash
pip install torch torchvision streamlit pandas scikit-learn kagglehub Pillow
