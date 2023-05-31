import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import os
import time

from dataset import NTU60_HRNET
from model import Model
from c3d_modified import C3dModifiedJointFramePersDim, C3dModifiedPersJointFrameDim

torch.cuda.empty_cache()

NUM_FRAMES_MIN = 32
NUM_PERS_MAX = 2
COORD_2D = 2
NUM_JOINTS = 17

batch_size = 10
num_epochs = 10

data_path = f'{os.getcwd()}/data/nturgbd/ntu60_hrnet.pkl'
label_map_path = f'{os.getcwd()}/tools/data/label_map/nturgbd_120.txt'

model_name = 'pjfd'
model_savepath = f'{os.getcwd()}/outputs/{model_name}/'

if model_name == 'jfpd':
  permute_order = (0, 3, 2, 1, 4)
  model = C3dModifiedJointFramePersDim()
  input_size = (batch_size, NUM_JOINTS, NUM_FRAMES_MIN, NUM_PERS_MAX, COORD_2D) 
elif model_name == 'pjfd':
  permute_order = (0, 1, 2, 3, 4)
  model = C3dModifiedPersJointFrameDim()
  input_size = (batch_size, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, COORD_2D)

train_dataset = NTU60_HRNET(data_path, label_map_path, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, permute_order, "train", 100)
val_dataset = NTU60_HRNET(data_path, label_map_path, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, permute_order, "val", 100)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Summary of the model
summary(model, input_size=input_size)

#Train model
TestModelDlav = Model(model)
start = time.perf_counter()
TestModelDlav.training(train_loader, val_loader, num_epochs, model_savepath)
stop = time.perf_counter()
print(f'The training is done in {stop - start} seconds')
