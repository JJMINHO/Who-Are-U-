# Who-Are-U: 150종 포켓몬 분류기 (Pokemon Classifier)

## 프로젝트 소개 (Project Overview)
주어진 포켓몬 이미지의 픽셀 특징을 분석하여 150종의 포켓몬 중 어떤 포켓몬인지 정확한 이름을 예측하는 컴퓨터 비전기반 분류기입니다.

PyTorch와 사전 Pre-trained CNN Backbone 모델을 활용한 **Transfer Learning** 기술을 적용했습니다. 이를 통해 처음부터 모델을 학습하는 것보다 훨씬 적은 컴퓨팅 자원과 시간으로 높은 분류 정확도를 제공합니다.

* **데이터셋 (Dataset):** Kaggle 7,000 Labeled Pokemon (Total 150 Classes)

---

## 실험 설정 및 성능 비교
본 프로젝트는 모델의 아키텍처와 전이 학습(Fine-tuning) 범위를 변수로 설정하여 총 4가지의 독립적인 실험을 진행했습니다.

1. **[실험 A] ResNet18 (Feature Extraction):**
ImageNet 데이터로 사전 학습된 가중치를 동결하고, 마지막 분류기(FC layer)만 150개 클래스에 맞게 학습했습니다.
3. **[실험 B] ResNet18 (Fine-Tuning):** 
사전 학습된 가중치를 초기값으로 사용하되, 전체 네트워크의 가중치를 포켓몬 데이터에 맞게 Fine-tuning했습니다.
4. **[실험 C] MobileNet V3 Small (Fine-Tuning):** 
ResNet 대비 파라미터 수가 약 1/7로 매우 적은 경량화 모델을 사용하여 Fine-Tuning을 진행했습니다.
5. **[실험 D] ResNet18 (From Scratch):** 
사전 학습 가중치를 전혀 사용하지 않고 무작위 초기화 상태에서 학습하여, 전이 학습의 효과를 대조군으로서 검증했습니다.

### 최종 성능 평가 표
> **Note:** 테스트셋을 분리하여 Precision, Recall, F1-Score를 측정

| 실험 | 설정 (Backbone & 방식) | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| A | ResNet18 (Feature Extraction) | 0.7522 | 0.7851 | 0.7630 | 0.7484 |
| B | ResNet18 (Fine-Tuning) | 0.8314 | 0.8695 | 0.8367 | 0.8315 |
| C | MobileNet V3 (Fine-Tuning) | 0.8695 | 0.8912 | 0.8698 | 0.8634 |
| D | ResNet18 (From Scratch) | 0.2302 | 0.2474 | 0.2366 | 0.1887 |

### Learning Curve
모델 학습 과정에서 Epoch에 따른 Loss의 감소와 Accuracy의 증가 추이입니다.

<img width="1507" height="669" alt="Image" src="https://github.com/user-attachments/assets/2e33b7c7-f3bc-4df9-babe-38a08ef151ed" />

**분석 요약:**
* **전이 학습(Transfer Learning)의 우수성:** 
사전 학습된 가중치를 사용한 모델 A, B, C는 초기 Epoch부터 빠르게 높은 정확도에 도달했습니다.
* **From Scratch 모델의 한계:** 
아무런 사전 정보 없이 학습을 시작한 모델 D는 동일한 Epoch 내에서 학습 속도가 매우 느리며 가장 낮은 성능을 기록했습니다.

---
## 데모 GUI (Streamlit Application)
사용자가 직접 포켓몬 이미지를 업로드하고 결과를 확인할 수 있는 인터랙티브 웹 데모를 구현했습니다. 
학습 결과 가장 높은 성능을 기록한 최종 채택 모델(`best_pokemon_model.pt`)의 아키텍처를 자동으로 인식하여 가중치를 로드하며, 분석 결과로 상위 5개의 예측 확률을 제공합니다.

---

### 추론 예시

#### 분류 성공 케이스
학습 데이터셋(150종)에 존재하는 포켓몬을 입력했을 때의 결과입니다. 
모델이 포켓몬의 시각적 특징을 정확하게 추출하여 높은 확률로 성공적인 분류를 수행했습니다.

<img width="600" alt="Success Case" src="https://github.com/user-attachments/assets/0c8bbc31-da3b-428b-a935-05f6528e26e7" />

<br>

#### 오분류 케이스
학습 데이터셋에 포함되지 않은 미학습 포켓몬을 입력한 경우입니다.
모델이 예측할 수 없는 새로운 범주의 데이터가 들어왔기 때문에, 모델의 한계로 인해 분류에 실패한 모습을 보여줍니다.

<img width="600" alt="Failure Case" src="https://github.com/user-attachments/assets/2b03fa7e-f1f4-43fc-9d15-6fc8e1bd5ecc" />
---

## 실행 방법 

프로젝트를 복제한 후, 아래 단계를 순서대로 진행하여 모델 학습 및 데모 앱을 실행할 수 있습니다.

### **1. 필수 패키지 설치**
프로젝트 구동에 필요한 라이브러리를 설치합니다.
```bash
pip install torch torchvision streamlit pandas scikit-learn kagglehub Pillow matplotlib
```

### **2. 모델 학습 및 성능 평가**
모델 학습을 실행
```bash
python training.py
```

### **3. 데모 웹 앱 실행**
```bash
streamlit run ui.py
```

*위 실행 방법은 사용자의 환경에 따라 달라질 수 있습니다.
