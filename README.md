# Emotion_detection
 This is an emotion detection application I made. I created my own convulational neural network(CNN) using pytorch and trained it on the FER2013 dataset. I managed to get a 64% testing set accuracy on the model which is fairly close to the human accuracy benchmark for this dataset (~65%). The application also uses the RESNET model to detect faces in the frame before the detected phases are passed to the CNN to detec emotions. OpenCV was also utilized to enable real-time emotion detection through a live video feed. 

 # Features
 - Real time emotion detection via webcam
 - Multi-face classification
 - Confidence scores for prediction

# Technology used
- Pytorch
- OpenCv
- ResNet(for face detection)
- Matplotlib

# Notes
- FER2013 is a noisy dataset so expect some misclassifcations
- The Dataset includes angry, surpirse, happy, sad, fear, neutral, disgust as emotions so the model will only detect those emotions.


# Example of Usage

https://github.com/user-attachments/assets/a445fba7-2462-4a9a-819f-d0301385af51




