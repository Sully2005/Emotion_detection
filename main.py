import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class CNET(nn.Module):
  def __init__(self):
    super().__init__()

    #First block
    self.conv1a = nn.Conv2d(1, 32, 3, padding=1)
    self.bn1a = nn.BatchNorm2d(32)
    self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
    self.bn1b = nn.BatchNorm2d(32)

    # Second block
    self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
    self.bn2a = nn.BatchNorm2d(64)
    self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
    self.bn2b = nn.BatchNorm2d(64)

    # Third block
    self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
    self.bn3a = nn.BatchNorm2d(128)
    self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
    self.bn3b = nn.BatchNorm2d(128)

    # Fourth block
    self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
    self.bn4a = nn.BatchNorm2d(256)
    self.conv4b = nn.Conv2d(256, 256, 3, padding=1)
    self.bn4b = nn.BatchNorm2d(256)


    #Fully connected layer
    self.fc1 = nn.Linear(256 * 3 * 3, 512)
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512, 256)
    self.dropout2 = nn.Dropout(0.5)
    self.fc3 = nn.Linear(256, 128)
    self.dropout3 = nn.Dropout(0.5)
    self.fc4 = nn.Linear(128, 7)

    self.pool = nn.MaxPool2d(2, 2)


    #Added softmax so I can display confidence
    self.softmax = nn.Softmax(1)




  def forward(self, x):
    #Block 1
    x = F.relu(self.bn1a(self.conv1a(x)))
    x = F.relu(self.bn1b(self.conv1b(x)))
    x = self.pool(x)


    #Block 2
    x = F.relu(self.bn2a(self.conv2a(x)))
    x = F.relu(self.bn2b(self.conv2b(x)))
    x = self.pool(x)

    #Block 3
    x = F.relu(self.bn3a(self.conv3a(x)))
    x = F.relu(self.bn3b(self.conv3b(x)))
    x = self.pool(x)

    #Block 4
    x = F.relu(self.bn4a(self.conv4a(x)))
    x = F.relu(self.bn4b(self.conv4b(x)))
    x= self.pool(x)


    #Fully connected
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.dropout1(x)
    x = F.relu(self.fc2(x))
    x = self.dropout2(x)
    x = F.relu(self.fc3(x))
    x = self.dropout3(x)
    x = self.fc4(x)
    x = self.softmax(x)



    return x




model_file = 'model.pth'

resnet_path = 'deploy.prototxt'
weight_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'

Resnet_model = cv2.dnn.readNetFromCaffe(resnet_path, weight_path)


loaded_model = CNET()

loaded_model.load_state_dict(torch.load(model_file, map_location='cpu'))

loaded_model.eval()

print(f"Model loaded succesfully from {model_file}")
print(f"Resnet model loaded successfully")




emotion_dictionary = {0: "Angry", 1: "Disgust", 2:"Fear", 3: "Happy", 4:"Sad", 5:"Surprise", 6: "Neutral"}

face_detecter = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()

  
  blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
  Resnet_model.setInput(blob)
  detections = Resnet_model.forward()
  grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #for each face detected
  for i in range(detections.shape[2]):
    confidence = detections[0,0, i, 2]

    if confidence > 0.5:
      #get box coordinates and upscale 
      box = detections[0,0,i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                                       frame.shape[1], frame.shape[0]])
      
      startx, starty, endx, endy = box.astype('int')
      grey_face_frame = grey[starty: endy, startx:endx]

      #Change Image to tensor, shape and normalize it like the testing set
      resized_image = cv2.resize(grey_face_frame, (48,48))
      input_tensor = torch.from_numpy(resized_image).float().unsqueeze(0).unsqueeze(0)
      input_tensor = (input_tensor / 255 - 0.5) / 0.5


      outputs = loaded_model(input_tensor)
      confidence, label = torch.max(outputs, 1)
      predicted_emotion = emotion_dictionary[label.item()]
      text = f"{predicted_emotion} ({confidence.item():.2f})"

      font = cv2.FONT_HERSHEY_TRIPLEX
      cv2.putText(frame, text, (startx, starty), font, 1, (0,255,0), 2)
      cv2.rectangle(frame, (startx, starty), (endx, endy), (0,255,0), 3)
      


  
  


  

  cv2.imshow("Emotion Detector", frame)

  #Press X to exit 
  if cv2.waitKey(1) == ord('x'):
    break
  
cap.release()
cv2.destroyAllWindows()




