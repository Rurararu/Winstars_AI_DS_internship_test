"""
Image Classification Inference - ResNet50
Simply run: python src/image_inference.py
(Change model_path and image_path in main())
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def predict(model_path, image_path):
    """Predict class for image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    img_size = checkpoint.get('img_size', 224)
    idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}

    # Model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Predict
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = idx_to_class[predicted.item()]
    return predicted_class


def main():
    model_path = '../models/image_classification/best_model.pth'
    image_path = '../data/test_data/pexels-jankoetsier-2647053.jpg'

    print(f"Model: {model_path}")
    print(f"Image: {image_path}\n")

    predicted_class = predict(model_path, image_path)

    print(f"Predicted class: {predicted_class}")
    return predicted_class


if __name__ == '__main__':
    main()