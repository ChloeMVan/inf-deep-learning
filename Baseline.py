import os
# -*- coding: utf-8 -*-
import argparse
import os
import urllib
import urllib.request
import tempfile
import time
import sys
import json
import random
from tqdm import tqdm
from os.path import join

# !ls FaceForensics/download.py
# !ls FaceForensics/dataset/extract_compressed_videos.py

# !sed -i 's|OUTPUT_BASE=~/Documents/INF-Deep_LearningFF_data|OUTPUT_BASE=FF_data|g' download_ff_data.sh

# # Verify
# !grep "OUTPUT_BASE" download_ff_data.sh

import subprocess
subprocess.run(['bash', 'download_ff_data.sh', '--num_videos', '1', '--frame_skip', '20'])

all_dirs = [
    '~/Documents/INF-Deep_Learning/FF_data/real/original_sequences/youtube/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/fake/manipulated_sequences/Deepfakes/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/DFD_real/original_sequences/actors/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/DFD_fake/manipulated_sequences/DeepFakeDetection/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/Face2Face/manipulated_sequences/Face2Face/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/FaceSwap/manipulated_sequences/FaceSwap/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/NeuralTextures/manipulated_sequences/NeuralTextures/c40/images'
]

for d in all_dirs:
    exists = os.path.exists(d)
    print(f"{'✓' if exists else '✗'} {d}")

import os

extractions = {
    '~/Documents/INF-Deep_Learning/FF_data/real/original_sequences/youtube/c40/videos':
        '~/Documents/INF-Deep_Learning/FF_data/real/original_sequences/youtube/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/DFD_real/original_sequences/actors/c40/videos':
        '~/Documents/INF-Deep_Learning/FF_data/DFD_real/original_sequences/actors/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/DFD_fake/manipulated_sequences/DeepFakeDetection/c40/videos':
        '~/Documents/INF-Deep_Learning/FF_data/DFD_fake/manipulated_sequences/DeepFakeDetection/c40/images',
    '~/Documents/INF-Deep_Learning/FF_data/NeuralTextures/manipulated_sequences/NeuralTextures/c40/videos':
        '~/Documents/INF-Deep_Learning/FF_data/NeuralTextures/manipulated_sequences/NeuralTextures/c40/images',
}

for video_dir, output_dir in extractions.items():
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_name = video_file.replace('.mp4', '')
            frame_output = os.path.join(output_dir, video_name)
            os.makedirs(frame_output, exist_ok=True)
            os.system(f"ffmpeg -i '{video_dir}/{video_file}' '{frame_output}/%04d.png' -hide_banner -loglevel error")
            print(f"✓ Extracted: {video_name}")

print("All done!")

for d in all_dirs:
    exists = os.path.exists(d)
    print(f"{'✓' if exists else '✗'} {d}")

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import pandas as pd
from collections import defaultdict

Image.MAX_IMAGE_PIXELS = None

def build_samples(all_dirs):
    samples = []

    if isinstance(all_dirs, dict):
        dirs_with_labels = all_dirs.items()
    else:
        dirs_with_labels = [
            (d, 0 if 'original' in d else 1)
            for d in all_dirs
        ]

    for directory, label in dirs_with_labels:
        for video_name in os.listdir(directory):
            video_path = os.path.join(directory, video_name)
            if not os.path.isdir(video_path):
                continue

            # Use full path as unique ID — guarantees no collisions
            unique_video_id = video_path

            for frame in sorted(os.listdir(video_path)):
                if frame.endswith('.png'):
                    frame_path = os.path.join(video_path, frame)
                    samples.append((frame_path, label, unique_video_id))

    print(f"Total frames: {len(samples)}")
    return samples

# split it by video
import pandas as pd
from sklearn.model_selection import train_test_split

def split_by_video(samples, test_size=0.2, random_state=42):
    """
    Splits samples into train and test by video name
    to prevent data leakage.

    Args:
        samples:      list of (frame_path, label, video_name)
        test_size:    proportion for test set (default 0.2)
        random_state: for reproducibility (default 42)

    Returns:
        train_samples, test_samples
    """
    all_videos = list(set(s[2] for s in samples))

    train_videos, test_videos = train_test_split(
        all_videos,
        test_size=test_size,
        random_state=random_state
    )

    train_samples = [s for s in samples if s[2] in train_videos]
    test_samples  = [s for s in samples if s[2] in test_videos]

    print(f"Total videos:  {len(all_videos)}")
    print(f"Train videos:  {len(train_videos)} | Train frames: {len(train_samples)}")
    print(f"Test videos:   {len(test_videos)}  | Test frames:  {len(test_samples)}")

    return train_samples, test_samples

# Build samples
all_samples = build_samples(all_dirs)

# Split by video
train_samples, test_samples = split_by_video(all_samples, test_size=0.2)

# dataset class
class BaselineDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, video_name = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, video_name

# Transform for ResNet-50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = BaselineDataset(train_samples, transform=transform)
test_dataset  = BaselineDataset(test_samples,  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# Quick check
for images, labels, video_names in train_loader:
    print(f"Batch shape: {images.shape}")  # should be (32, 3, 224, 224)
    print(f"Labels: {labels}")
    break

train_dataset = BaselineDataset(train_samples, transform=transform)
test_dataset  = BaselineDataset(test_samples,  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# Quick check
for images, labels, video_names in train_loader:
    print(f"Batch shape: {images.shape}")  # should be (32, 3, 224, 224)
    print(f"Labels: {labels}")
    break

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import defaultdict
from PIL import Image
import pandas as pd
import numpy as np
import os

# load model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Load pretrained ResNet-50
model = models.resnet50(pretrained=True)

# Replace final layer for binary classification (real vs fake)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

print("Model loaded!")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

import os

save_path = '/content/resnet50_baseline.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('/content/resnet50_baseline.pth',
                                  map_location=device))
model = model.to(device)
model.eval()
print("Model loaded!")

from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, f1_score

model.eval()

video_predictions = defaultdict(list)
video_true_labels = {}

with torch.no_grad():
    for images, labels, video_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for video_name, pred, label in zip(video_names, preds, labels.numpy()):
            video_predictions[video_name].append(int(pred))
            video_true_labels[video_name] = int(label)

# Apply majority voting
y_true = []
y_pred = []

for video_name, frame_preds in video_predictions.items():
    fake_votes  = sum(frame_preds)
    total_votes = len(frame_preds)
    final_pred  = 1 if fake_votes > total_votes / 2 else 0
    y_true.append(video_true_labels[video_name])
    y_pred.append(final_pred)

print("\n=== BASELINE RESULTS (Video Level) ===\n")

# Check what's in y_true and y_pred
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"Unique classes in y_true: {set(y_true)}")
print(f"Unique classes in y_pred: {set(y_pred)}")

print(classification_report(
    y_true, y_pred,
    labels=[0, 1],                              # force both classes
    target_names=['Real (0)', 'Fake (1)'],
    zero_division=0                             # handle missing classes gracefully
))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print(f"\nF1 Score (weighted): {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score (macro):    {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")




print("\n=== BASELINE RESULTS (Video Level) ===\n")
print(classification_report(
    y_true, y_pred,
    target_names=['Real (0)', 'Fake (1)']
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print()
print(f"F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
print(f"F1 Score (macro):    {f1_score(y_true, y_pred, average='macro'):.4f}")


