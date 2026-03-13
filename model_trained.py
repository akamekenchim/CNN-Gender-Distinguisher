from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size = 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Dropout(0.2)
    )
    self.classifier = nn.Sequential(
        nn.Linear(128*28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )
  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])
transformer2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])
print("Evaluation - starting...")
model = CNN()
model.load_state_dict(torch.load("cnn_gender_model_5.pth", map_location=torch.device('cpu')))
model.eval()

all_data = ImageFolder(root="test_images", transform = transformer2)
all_data_loader = DataLoader(all_data, batch_size = 1, shuffle = False)
for i, (images, labels) in enumerate(all_data_loader):
  with torch.no_grad(): # Tối ưu bộ nhớ khi dự đoán
      output_final = model(images)
      # Lấy chỉ số của lớp có giá trị cao nhất
      du_doan = torch.argmax(output_final, dim=1).item()
  img_path = all_data.samples[i][0].split('/')[-1]
  print(f"Ảnh:---{img_path}---")
  print(f"Logits: {output_final}")
  classes_name = ["Nam", "Nữ"]
  print(f"Dự đoán: {du_doan}, đó là {classes_name[du_doan]}")

