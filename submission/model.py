import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os
import cv2
from os import listdir



torch.set_grad_enabled(True)
class Model():
    def __init__(self, model):
        # instantiate model + optimizer + loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.005)
        self.criterion = nn.MSELoss()
        self.output = None

    def load_pretrained_model(self, model_savepath):
        # This loads the parameters saved in bestmodel.pth into the model
        checkpoint = torch.load(model_savepath, map_location=self.device)
        self.model.load_state_dict(checkpoint)

    def train(self, loader):
        self.model.training = True

        full_outputs = []
        full_labels = []
        losses = []
        for batch in tqdm(loader):
            skeletons, labels, _ = batch
            labels_one_hot = F.one_hot(labels, 60).float()
            pred = self.model(skeletons)
            loss = self.criterion(pred, labels_one_hot)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            full_outputs.append(pred)
            full_labels.append(labels)
            losses.append(loss)

        full_outputs = torch.cat(full_outputs).cpu()
        full_labels = torch.cat(full_labels).cpu()
        losses = torch.stack(losses).mean().cpu()

        acc = self.top1_accuracy(full_outputs.detach().numpy(), full_labels.numpy())
        return acc, full_outputs.detach().numpy(), full_labels.numpy(), losses
		
    @torch.no_grad()
    def validate(self, loader):
        self.model.training = True

        full_outputs = []
        full_labels = []
        losses = []
        for batch in tqdm(loader):
            skeletons, labels, _ = batch
            labels_one_hot = F.one_hot(labels, 60).float()
            pred = self.model(skeletons)
            loss = self.criterion(pred, labels_one_hot)
            full_outputs.append(pred)
            full_labels.append(labels)
            losses.append(loss)

        full_outputs = torch.cat(full_outputs).cpu()
        full_labels = torch.cat(full_labels).cpu()
        losses = torch.stack(losses).mean().cpu()

        acc = self.top1_accuracy(full_outputs.detach().numpy(), full_labels.numpy())
        return acc, full_outputs.detach().numpy(), full_labels.numpy(), losses
	
    def training(self, train_loader, val_loader, nb_epochs, model_savepath):
        
        if not os.path.exists(model_savepath):
          os.makedirs(model_savepath)

        epochs = nb_epochs
        best_acc = 0

        list_train_acc = []
        list_train_loss = []
        list_val_acc = []
        list_val_loss = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # Train
            train_acc, train_outputs, train_labels, train_loss = self.train(train_loader)
            top5_acc = self.top5_accuracy(train_outputs, train_labels)
            print(f"training metrics : Top-1 accuracy = {train_acc} Top-5 accuracy = {top5_acc} loss = {train_loss}")
            list_train_acc.append(train_acc)
            list_train_loss.append(train_loss.detach().numpy())

            # Evaluate
            val_acc, val_outputs, val_labels, val_loss = self.validate(val_loader)
            top5_acc = self.top5_accuracy(val_outputs, val_labels)
            print(f"validation metrics : Top-1 accuracy = {val_acc} Top-5 accuracy = {top5_acc} loss = {val_loss}")
            list_val_acc.append(val_acc)
            list_val_loss.append(val_loss.detach().numpy())

            # Save the model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(model_savepath)

        self.plot_metrics(val_outputs, val_labels, model_savepath)

        np.save(model_savepath+'train_acc.npy', list_train_acc)
        np.save(model_savepath+'train_loss.npy', list_train_loss)
        np.save(model_savepath+'val_acc.npy', list_val_acc)
        np.save(model_savepath+'val_loss.npy', list_val_loss)

        fig, axs = plt.subplots(2,1, figsize=(20,10))

        axs[0].plot(list_train_acc, label='train')
        axs[0].plot(list_val_acc, label='validation')
        axs[0].legend()
        axs[0].set_title('Accuracy')
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Accuracy")

        axs[1].plot(list_train_loss, label='train')
        axs[1].plot(list_val_loss, label='validation')
        axs[1].legend()
        axs[1].set_title('Loss')
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("MSE")

        plt.savefig(model_savepath + "fig_train_skeleton.png")

    def save_model(self, model_savepath):
        torch.save(self.model.state_dict(), model_savepath + 'bestmodel.pth')
	
    @torch.no_grad()
    def predict(self, loader):
        self.model.training = False

        full_outputs = []
        full_labels = []
        losses = []
        full_names = []
        for batch in tqdm(loader):
            skeletons, labels, names = batch
            labels_one_hot = F.one_hot(labels, 60).float()
            pred = self.model(skeletons)
            loss = self.criterion(pred, labels_one_hot)
            full_outputs.append(pred)
            full_labels.append(labels)
            losses.append(loss)
            full_names = np.concatenate((full_names, names))

        full_outputs = torch.cat(full_outputs).cpu()
        full_labels = torch.cat(full_labels).cpu()
        losses = torch.stack(losses).mean().cpu()

        acc = self.top1_accuracy(full_outputs.detach().numpy(), full_labels.numpy())
        return acc, full_outputs.detach().numpy(), full_labels.numpy(), losses.detach().numpy(), full_names
		
    def show_prediction(self, skeletons, output, label, name, label_dict, permute_order, prediction_savepath, video_path):
		
        permute_back_order = [0,0,0,0]
        for i in range(4):
            permute_back_order[permute_order[i+1]-1] = i
        skeletons = np.transpose(skeletons, permute_back_order)

        size = (1920, 1080)
        font = cv2.FONT_HERSHEY_SIMPLEX
        video_flag = False
        vid_length = skeletons.shape[1]
        fps = 15

        ind_select = np.flip(np.argsort(output))
        n = 0
        best_matchs = []

        for i in ind_select:
            if (output[i] > 5e-2) and (n <= 5):
                best_matchs.append(i)
                n += 1

        for f in listdir(video_path):
            if f.split('.')[0] == name+'_rgb':
                video_flag = True
                filepath = video_path + f
                fps = 25

        if video_flag:
            cam = cv2.VideoCapture(filepath)
            vid_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(prediction_savepath + name + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        list_ind = np.linspace(0, vid_length, num=skeletons.shape[1], endpoint=False, dtype=int)

        for i in range(vid_length):
            if video_flag:
                ret,img = cam.read()
            else:
                img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

            ind = np.argwhere(list_ind <= i)[-1][0]

            for j in range(skeletons.shape[0]):
                color = (255*((j+1)%2), 0, 255*(j%2))
                for k in range(skeletons.shape[2]):
                    tmp_skull = (int(skeletons[j][ind][k][0]*size[0]),int(skeletons[j][ind][k][1]*size[1]))
                    if tmp_skull == (0,0):
                        continue
                    cv2.circle(img, tmp_skull, 5, color, -1)

            cv2.putText(img,f'True value: {label_dict[label]}',(10,50), font, 1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(img,f'True value: {label_dict[label]}',(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)

            for j in range(len(best_matchs)):
                if best_matchs[j] == label:
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                cv2.putText(img,f'Pred: {label_dict[best_matchs[j]]} : {round(output[best_matchs[j]], 2)}',(10,50*(j+2)), font, 1, (255,255,255),2,cv2.LINE_AA)
                cv2.putText(img,f'Pred: {label_dict[best_matchs[j]]} : {round(output[best_matchs[j]], 2)}',(10,50*(j+2)), font, 1, color,1,cv2.LINE_AA)

            out.write(img)

        out.release()
        if video_flag:
            cam.release()
		

    def top1_accuracy(self, outputs, labels):
        predictions = np.argmax(outputs, axis=1)
        return np.sum(predictions == labels)/len(outputs)

    def top5_accuracy(self, outputs, labels):

        ind_select = np.flip(np.argsort(outputs))

        count = 0

        for i in range(ind_select.shape[0]):
            n=0
            for j in range(ind_select.shape[1]):
                if (outputs[i][ind_select[i][j]] > 5e-5) and (n <= 5):
                    n += 1
                    if j == labels[i]:
                        count+=1
                        break
            
        return count/ind_select.shape[0]

    def plot_metrics(self, outputs, labels, model_savepath):

        predictions = np.argmax(outputs, axis=1)
        cm = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(include_values=False)
        disp.ax_.set_xticks(np.arange(0,60,5))
        disp.ax_.set_yticks(np.arange(0,60,5))
        plt.savefig(model_savepath + "fig_train_confusion_matrix.png")
        print(classification_report(labels, predictions, zero_division=0))

        