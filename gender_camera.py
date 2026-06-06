import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

size = 128*28*28

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features= size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2)
        )   
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
transformer2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = myCNN().to(device)
model.load_state_dict(torch.load("cnn_gender_model_5.pth", map_location=device))
model.eval()


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if not cap.isOpened():
    print("Cannot access the Camera")
    exit()

while True:
    ret, frame = cap.read()
    if ret is not True:
        print("Lost connection")
        break
    
    frame = cv2.resize(frame, (800, 540), cv2.INTER_AREA)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        image=grayFrame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for f in faces:
        hair_delta = 20
        x, y, w, h = f
        
        #max_y, max_x = frame.shape[0], frame.shape[1]
        #y_start = max(0, y - int(hair_delta * 2))
        #y_end   = min(max_y, y + h)
        #x_start = max(0, x - int(hair_delta))
        #x_end   = min(max_x, x + w + int(hair_delta))
        #face_cropped = frame[y_start:y_end, x_start:x_end]
        face_cropped = frame[y:y+h, x:x+w]
        if face_cropped.size == 0:
            continue
        face_rgb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)
    
        face_pil = Image.fromarray(face_rgb)
        face_tensor = transformer2(face_pil)
        
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(face_tensor) 
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
            
        label = "Nam" if predicted_class == 0 else "Nu"
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Cua so", frame)


    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()