import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import kagglehub
import json
import os

# 모델 정의 함수 (4가지 실험 설정)
def get_model(experiment_type, num_classes=150):
    if experiment_type == 'A':
        # 실험 A: ResNet18 (Feature Extraction - 가중치 동결)
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif experiment_type == 'B':
        # 실험 B: ResNet18 (Fine-tuning - 전체 학습)
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif experiment_type == 'C':
        # 실험 C: MobileNet V3 Small (다른 Backbone 파인 튜닝)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    elif experiment_type == 'D':
        # 실험 D: ResNet18 (From Scratch - 사전 학습 X)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# 2. 모델 검증 함수 (Precision, Recall, F1 계산)
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # zero_division=0 은 분모가 0일 때 경고를 없애줍니다.
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    return acc, precision, recall, f1

# 3. Learning 및 비교 루프
def train_and_compare():
    print("1. Kaggle 데이터셋 다운로드 중...")
    dataset_path = kagglehub.dataset_download("lantian773030/pokemonclassification")

    data_dir = dataset_path
    if os.path.exists(os.path.join(dataset_path, "PokemonData")):
        data_dir = os.path.join(dataset_path, "PokemonData")

    print("\n2. 데이터 전처리 (ImageNet 규격 적용)...")
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    class_names = full_dataset.classes
    with open('pokemon_classes.json', 'w') as f:
        json.dump(class_names, f)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    experiments = {
        'A': 'ResNet18 (Feature Extraction)',
        'B': 'ResNet18 (Fine-Tuning)',
        'C': 'MobileNet V3 (Fine-Tuning)',
        'D': 'ResNet18 (From Scratch)'
    }
    results = {}
    num_epochs = 3  # 빠른 결과를 위해 3으로 설정. 시간 여유가 있다면 5나 10으로 올려보세요.

    print(f"\n3. 총 4가지 실험 자동 비교를 시작합니다. (Device: {device})")
    print("-" * 50)

    best_acc = 0.0

    for exp_type, exp_desc in experiments.items():
        print(f"🚀 [실험 {exp_type}] {exp_desc} 학습 중...")
        model = get_model(exp_type, num_classes=len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss()

        if exp_type == 'A':
            optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 모델 학습
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 검증셋 평가
        print(f"📊 [실험 {exp_type}] 성능 평가 중...")
        acc, prec, rec, f1 = evaluate_model(model, val_loader, device)
        results[exp_type] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

        print(f"결과 -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}\n")

        # 가장 성능이 좋은 모델 저장
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_pokemon_model.pt')

    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("🏆 [최종 실험 결과 요약] - README.md에 복사하세요!")
    print("=" * 50)
    print("| 실험 | 설정 (Backbone & 방식) | Accuracy | Precision | Recall | F1-Score |")
    print("|---|---|---|---|---|---|")
    for exp_type, metrics in results.items():
        desc = experiments[exp_type]
        print(
            f"| {exp_type} | {desc} | {metrics['Accuracy']:.4f} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} | {metrics['F1-Score']:.4f} |")
    print("=" * 50)
    print("✅ 최고 성능 모델이 'best_pokemon_model.pt'로 저장되었습니다.")

if __name__ == '__main__':
    train_and_compare()