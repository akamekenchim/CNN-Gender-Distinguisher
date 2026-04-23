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
      nn.Dropout(0.28)
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
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(0.13),
    transforms.RandomPerspective(0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])
transformer2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device) # Đưa model lên GPU
#thay bằng đường dẫn đến file .pth bạn muốn load
model.load_state_dict(torch.load("/content/drive/MyDrive/Nhập môn AI - Model Training/gender_savestate/cnn_gender_model_5.pth"))
print(f"Training started on {device}...")

data = ImageFolder(root="/content/temp_data", transform = transformer)
datasets = DataLoader(data, batch_size = 8, shuffle = True)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
print("Training started...")
for i in range (80):
  total_loss = 0
  for images, labels in datasets:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  avg_loss = total_loss / len(datasets)
  scheduler.step(avg_loss)
  if((i+1) % 1 == 0) :
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Đã học xong lần {i+1}")
    print(f"Loss TB: {avg_loss:.4f} - Loss rate: {current_lr}")


model.eval()
# Lưu lại bộ não AI
# thay bằng đường dẫn đến thư mục bạn muốn lưu trạng thái model
torch.save(model.state_dict(), "/content/drive/MyDrive/Nhập môn AI - Model Training/gender_savestate/cnn_gender_model_8.pth") 

print("Đã lưu mô hình thành công! 🎉")
