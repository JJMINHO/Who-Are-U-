import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import os
import pandas as pd

# 클래스 이름 로드
try:
    with open('pokemon_classes.json', 'r') as f:
        class_names = json.load(f)
except FileNotFoundError:
    class_names = [f"Pokemon_{i}" for i in range(150)]


# 모델 로드 함수 
@st.cache_resource
def load_model():
    num_classes = len(class_names)
    is_loaded = False

    if os.path.exists('best_pokemon_model.pt'):
        state_dict = torch.load('best_pokemon_model.pt', map_location='cpu')

        if 'classifier.3.weight' in state_dict:
            model = models.mobilenet_v3_small(weights=None)
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, num_classes)
            model_name = "MobileNet V3 Small"
        else:
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            model_name = "ResNet18"

        model.load_state_dict(state_dict)
        is_loaded = True
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_name = "ResNet18 (임시)"

    model.eval()
    return model, is_loaded, model_name


# 앱 구동 시 모델 로드
model, is_model_loaded, loaded_model_name = load_model()

if is_model_loaded:
    st.toast(f"✅ 최고 성능 모델({loaded_model_name})을 성공적으로 불러왔습니다!", icon="🔥")
else:
    st.warning("⚠️ 'best_pokemon_model.pt' 파일이 없습니다. 임시 모델로 실행됩니다.")


# 이미지 전처리 설정 (ImageNet 규격)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Streamlit UI 구성
st.title("Who Are U? : Pokemon Classifier")
st.write(f"현재 최고 성능 모델: **{loaded_model_name}**")
st.write("이미지를 업로드하면 어떤 포켓몬인지 분석해 드립니다!")

uploaded_file = st.file_uploader("포켓몬 이미지를 업로드해 주세요...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_container_width=True)

    with st.spinner("분석 중입니다... 🔍"):
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("Top-5 Predictions:")
    for i in range(top5_prob.size(0)):
        class_index = top5_catid[i].item()
        prob = top5_prob[i].item() * 100
        pokemon_name = class_names[class_index]
        st.write(f"{i + 1}. **{pokemon_name}** ({prob:.2f}%)")


# 모델별 성능 및 추론 능력 비교표 
st.markdown("---")
st.subheader("모델별 성능 및 추론 능력 비교")

data = {
    "실험 (모델 설정)": [
        "A. ResNet18 (Feature Extraction)",
        "B. ResNet18 (Fine-Tuning)",
        "C. MobileNet V3 Small (Fine-Tuning)",
        "D. ResNet18 (From Scratch)"
    ],
    "파라미터 수 (크기)": ["약 11M (무거움)", "약 11M (무거움)", "약 1.5M (가벼움)", "약 11M (무거움)"],
    "Accuracy": [0.7522, 0.8314, 0.8695, 0.2302],
    "Precision": [0.7851, 0.8695, 0.8912, 0.2474],
    "Recall": [0.7630, 0.8367, 0.8698, 0.2366],
    "F1-Score": [0.7484, 0.8315, 0.8634, 0.1887]
}

df = pd.DataFrame(data)

# 데이터프레임을 Streamlit에 출력
st.dataframe(df, use_container_width=True)

st.info("""
**분석 결과 요약**
* **Pre-trained 모델의 위력:** 아무것도 모르는 상태에서 학습한(From Scratch) 모델 D보다 사전 학습 가중치를 사용한 모델들의 성능이 압도적으로 높습니다.
* **가벼운 모델의 효율성:** MobileNet V3는 파라미터 수가 ResNet18의 약 1/7 수준으로 매우 가벼워 모바일 기기에서도 빠른 추론이 가능하면서도, 준수한 성능을 보여줍니다.
""")
