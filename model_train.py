#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from classifier_model import AIClassifier
import sklearn
import numpy as np


# In[14]:


import subprocess
subprocess.run(['bash', 'download_ff_data.sh', '--num_videos', '10', '--frame_skip', '20'])

# In[15]:


"""
Prereqs: download 10 vids/directory in all_dirs via download_ff_data.sh
"""
def group_by_video(directory):
    groups = defaultdict(list)
    for video_name in os.listdir(directory):
        video_path = os.path.join(directory, video_name)
        if not os.path.isdir(video_path):
            continue
        frames = sorted(f for f in os.listdir(video_path) if f.endswith('.png'))
        if frames:
            # change: doesn't prepend video name to frame filename
            groups[video_name] = [f for f in frames]
    
    result = []
    for key, frames in groups.items():
        result.append([key, frames])
    return result


# In[21]:


all_dirs = ['FF_data/real/original_sequences/youtube/c40/images', 
            'FF_data/fake/manipulated_sequences/Deepfakes/c40/images',
            'FF_data/DFD_real/original_sequences/actors/c40/images',
            'FF_data/DFD_fake/manipulated_sequences/DeepFakeDetection/c40/images',
            'FF_data/Face2Face/manipulated_sequences/Face2Face/c40/images',
            'FF_data/FaceSwap/manipulated_sequences/FaceSwap/c40/images',
            'FF_data/NeuralTextures/manipulated_sequences/NeuralTextures/c40/images'
            ]

'''
0 - Authentic
1 - Deepfake
'''
# same thing as the original version, but combines the root dir and vidpath to
# make ID'ing unique vids easier
# skip is the increment of frames btwn frames in a clip
def create_datamap(all_dirs, frames_per_clip=10, skip=1):
    data = {
        'video_path': []
    }
    for i in range(10):
        data['frame_' + str(i)] = []
    
    data['label'] = []

    for dir in all_dirs:
        label = 1
        if 'original' in dir:
            label = 0
        # list of videos in dir, along with a list of frame filenames with them 
        #[[vid, [vid/fr1.png, vid/fr2.png, ...]], ...]
        video_map = group_by_video(dir)
        for video, frames in video_map:
            i = 0
            for j in range(frames_per_clip * skip, len(frames), frames_per_clip * skip):
                if skip == 1:
                    split = frames[i:j]
                else:
                    split = [frames[idx] for idx in range(i, j, skip)]
                i = j
                data['video_path'].append(os.path.join(dir, video))
                data['label'].append(label)
                for k in range(len(split)):
                    data['frame_' + str(k)].append(split[k])
    return data

data_map = create_datamap(all_dirs, skip=1)
df = pd.DataFrame(data_map)
df.to_csv('dataset_splits_skip1.csv', index=False) # for testing dataset


# In[22]:


df = pd.read_csv("dataset_splits_skip1.csv")
print(f"Total frames = {len(df)}")
vidnames = list(set(df["video_path"]))
print(f"Number of videos (should be 70): {len(vidnames)}") 
real_vids = [name for name in vidnames if "original" in name]
fake_vids = [name for name in vidnames if "original" not in name]
print(len(real_vids), len(fake_vids))

real_train, real_test = train_test_split(real_vids, test_size=0.1)
fake_train, fake_test = train_test_split(fake_vids, test_size=0.1)
train_vids = real_train + fake_train
# TEMP
train_vids = train_vids[:500]
test_vids = real_test + fake_test
train_frames = df.loc[[vid in train_vids for vid in df["video_path"]]]
test_frames = df.loc[[vid in test_vids for vid in df["video_path"]]]


# In[23]:


if not torch.cuda.is_available():
    print("!!!\n\tCUDA not available. Everything ok?\n!!!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class FrameDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.df)
    
    def get_labels(self):
        return self.df["label"]

    def __getitem__(self, idx):
        """
        Returns a clip tensor of shape (3, T, H, W) = (3, 10, 500, 500).
        The DataLoader will stack these into (B, 3, T, H, W).
        """
        vidpath = self.df.iloc[idx, 0]
        tensors = []
        for i in range(10):
            fname = self.df.iloc[idx, i + 1]  # cols 1–10 are frame_0–frame_9
            path = os.path.join(vidpath, fname)
            img = Image.open(path).convert('RGB').resize((500, 500))
            tensors.append(self.transform(img)) # (3, 500, 500)
        # TODO normalize pixel vals
        clip = torch.stack(tensors, dim=1).to(dtype=torch.float32) # (3, T, 500, 500)
        label = torch.Tensor([self.df.iloc[idx, -1]])
        return clip, label


train_dataset = FrameDataset(train_frames)
train_loader = DataLoader(train_dataset, batch_size=8)
test_dataset = FrameDataset(test_frames)


# In[24]:


def test_model(model, data, device='cpu', do_print=False):
    model.to(device=device)
    model.eval()
    total = 0
    correct = 0
    labels = ["Real", "Fake"]
    record = []
    out_text = []
    for i, (frames, label) in enumerate(data):
        frames = frames.unsqueeze(0).to(device) # add batch dimension
        output = model(frames)[0] # since batch size is 1, we will get only one output
        y_pred = torch.round(output).item()
        y_actual = label
        record.append((y_pred, y_actual))
        if y_actual == y_pred:
            correct = correct+1
        total = total+1
    out_text.append(f"Accuracy = {correct / total * 100:.4f} %")
    for i in range(2):
        tps = sum([1 for x in record if x[0] == i and x[1] == i])
        fps = sum([1 for x in record if x[0] == i and x[1] != i])
        fns = sum([1 for x in record if x[0] != i and x[1] == i])
        tns = sum([1 for x in record if x[0] != i and x[1] != i])
        guesses = [x[0] for x in record if x[1] == i]
        # represents what the model guessed when this was the actual label
        counts = [guesses.count(j) for j in range(2)]
        if tps == 0:
            precision = 0.0
            recall = 0.0
            f_score = 0.0
        else:
            precision = tps / (tps + fps)
            recall = tps / (tps + fns)
            f_score = 2 * precision * recall / (precision + recall)
        out_text.append(f"  {labels[i]} ({i}):\n    F Score: {f_score:.4f}\n    Guesses: {counts}\n    Precision: {precision:.4f}\n    Recall: {recall:.4f}")
        if i == 1:
            out_text.append(f"\tPred Pos.\tPred Neg.\n\tReal Pos.\t{tps}\t{fns}\n\tReal Neg.\t{fps}\t{tns}")
    out_text = "\n".join(out_text)
    if do_print:
        print(out_text)
    return correct / total * 100, out_text


# In[7]:


import pickle
from tqdm import tqdm
def train_model(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    num_epochs, 
    test_data, 
    output_dir, 
    start_epoch=0, 
    losses=[], 
    vals=[], 
    device='cpu', 
    save_every=50
):
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    model.to(device)
    best_val = vals.max() if len(vals) > 0 else -1
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} / {num_epochs}")
        for frames, labels in progress_bar:
            # Moving the data to GPU if available
            frames, labels = frames.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            losses.append(loss.item())
        val, output = test_model(model, test_data, device)
        vals.append(val)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Val: {val}")

        with open(os.path.join(output_dir, "stats.pkl"), "wb") as f:
            pickle.dump({"epoch": epoch + 1, "loss": losses, "val": vals}, f)
        
        if val > best_val:
            print(f" ! New best val: {val:.4f} !\nSaving to epc_{epoch + 1:04d}_val_{val:.3f}.pkl")
            best_val = val
            print(output)
            torch.save(
                {"epoch": epoch + 1, "losses": losses, "vals": vals, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                os.path.join(output_dir, "checkpoints", f"epc_{epoch + 1:04d}_val_{val:.3f}.pkl")
            )
        elif (epoch + 1) % save_every == 0:
            print(f"Checkpoint epoch: saving to epc_{epoch + 1:04d}_val_{val:.3f}.pkl")
            torch.save(
                {"epoch": epoch + 1, "losses": losses, "vals": vals, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, 
                os.path.join(output_dir, "checkpoints", f"epch_{epoch + 1:04d}_val_{val:.3f}.pkl")
            )
    return losses, vals


# In[8]:


# weight classes to account for imbalance
weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(train_dataset.get_labels()), y=train_dataset.get_labels())
pos_weight = torch.Tensor([weights[1] / weights[0]])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion.to(device=device)


# In[9]:


lr = .0001
epochs = 500
output_dir = "output-tenfiles"

model = AIClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.to(device)
# set None for fresh training, file name in the checkpoints directory to load
saved_model = None
if saved_model:
    chkpt = torch.load(os.path.join(output_dir, "checkpoints", saved_model))
    model.load_state_dict(chkpt["model"])
    optimizer.load_state_dict(chkpt["optimizer"])
    start = chkpt["epoch"]
    losses = chkpt["losses"]
    vals = chkpt["vals"]
else:
    start = 0
    losses = []
    vals = []
    
losses, vals = train_model(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    epochs, 
    test_dataset, 
    output_dir, 
    start_epoch=start, 
    losses=losses, 
    vals=vals, 
    device=device
)


# In[ ]:


with open("output-tenfiles/checkpoints/epc_0001_val_67.647", "r") as f:
    stuff = torch.load(f)
print(stuff)


# In[ ]:




