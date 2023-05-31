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

batch_size = 50


data_path = f'{os.getcwd()}/data/nturgbd/ntu60_hrnet.pkl'
label_map_path = f'{os.getcwd()}/tools/data/label_map/nturgbd_120.txt'
video_path = f'{os.getcwd()}/ntu_samples/'

model_name = 'jfpd'
model_savepath = f'{os.getcwd()}/outputs/{model_name}/'

if model_name == 'jfpd':
  permute_order = (0, 3, 2, 1, 4)
  model = C3dModifiedJointFramePersDim()
  input_size = (batch_size, NUM_JOINTS, NUM_FRAMES_MIN, NUM_PERS_MAX, COORD_2D) 
elif model_name == 'pjfd':
  permute_order = (0, 1, 2, 3, 4)
  model = C3dModifiedPersJointFrameDim()
  input_size = (batch_size, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, COORD_2D)


test_dataset = NTU60_HRNET(data_path, label_map_path, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, permute_order, "val")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

TestModelDlav = Model(model)
TestModelDlav.load_pretrained_model(model_savepath +  "bestmodel.pth")
start = time.perf_counter()
acc, full_outputs, full_labels, loss, full_names = TestModelDlav.predict(test_loader)
stop = time.perf_counter()
print(f'The prediction is done in {stop - start} seconds for {len(test_dataset)} predictions.')

print(f"testing metrics : accuracy = {acc}  loss = {loss}")

find_flag = False
for f in os.listdir(video_path):
  videoname = f.split('.')[0].split('_')[0]
  print(videoname)
  for i in range(len(test_dataset)):
    if test_dataset.names[i] == videoname:
      idx=i
      find_flag = True
      print(idx)
      break
  if find_flag:
    break

TestModelDlav.show_prediction(test_dataset.skeletons[idx].cpu().numpy(), full_outputs[idx], full_labels[idx], full_names[idx], test_dataset.label_dict, permute_order, model_savepath, video_path)
