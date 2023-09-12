import os
import numpy as np
import pandas as pd
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
#from pythreading import Thread
import requests
import time as t


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

# Load the saved model
model_path = r'D:\My Files\BUET 1906033\L3T2\EEE 318\Projects\Dataset\entire_model_saved.pt'
model = torch.load(model_path, map_location = torch.device('cpu'))

# Preprocess your input data
input_data_path = r'C:\Users\User\Desktop\Test Dataset\bottle.jpeg'  # Your input data path (e.g., an image)
input_data = cv2.imread(input_data_path, -1)
input_data = input_data[...,::-1]
plt.figure(figsize = (5, 5))
plt.imshow(input_data)
# Apply the same preprocessing as during training (e.g., normalization)
input_data = torchvision.transforms.ToPILImage()(input_data)
input_data = torchvision.transforms.Resize((256, 256))(input_data)
input_data = torchvision.transforms.ToTensor()(input_data)


with torch.no_grad():
    # Forward pass
    output = model(input_data.unsqueeze(0))  # Add a batch dimension (assuming input_data is a single sample)
    # Convert the output to probabilities or class predictions
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

print(predicted_class)
#cardboard = 0, glass = 1, metal = 2, paper = 3, plastic = 4, trash = 5
label2name = {0 : 'cardboard', 1 : 'glass', 2 : 'metal' , 3 : 'paper', 4 : 'plastic', 5 : 'trash'}
print(label2name[predicted_class])

"""
class RTSPVideoWriterObject(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.output_video = cv2.VideoWriter('output.avi', self.codec, 30, (self.frame_width, self.frame_height))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow('frame', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)

if __name__ == '__main__':
    rtsp_stream_link = 'http://192.168.4.1:81/stream'
    video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link)
    while True:
        try:
            video_stream_widget.show_frame()
            #video_stream_widget.save_frame()
        except AttributeError:
            pass
"""

LiveVideo = VideoStream(src='http://192.168.4.1:81/stream').start()
while True:
    frame = LiveVideo.read()
    timestr = t.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite("frame%s.jpg" % timestr, frame )
