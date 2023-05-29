import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2



torch.set_grad_enabled(True)
class Model():
    def __init__(self, model):
        # instantiate model + optimizer + loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device', self.device)
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

        acc = self.accuracy(full_outputs, full_labels)
        return acc, full_outputs, full_labels, losses
		
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

        acc = self.accuracy(full_outputs, full_labels)
        return acc, full_outputs, full_labels, losses
	
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
            train_acc, _, _, train_loss = self.train(train_loader)
            print(f"training metrics : accuracy = {train_acc}  loss = {train_loss}")
            list_train_acc.append(train_acc)
            list_train_loss.append(train_loss.detach().numpy())

            # Evaluate
            val_acc, _, _, val_loss = self.validate(val_loader)
            print(f"validation metrics : accuracy = {val_acc}  loss = {val_loss}")
            list_val_acc.append(val_acc)
            list_val_loss.append(val_loss.detach().numpy())

            # Save the model
            if val_acc > best_acc:
                best_acc = val_acc
				self.save_model(model_savepath)

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
        plt.show()

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

        acc = self.accuracy(full_outputs, full_labels)
        return acc, full_outputs.detach().numpy(), full_labels.numpy(), losses.detach().numpy(), full_names
		
	def show_prediction(skeletons, output, label, name, label_dict, prediction_savepath, video_path):
	
		size = (1920, 1080)
		font = cv2.FONT_HERSHEY_SIMPLEX
	
		out = cv2.VideoWriter(prediction_savepath + name + 'avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
		
		for i in range(skeletons.shape[1]):
			img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
			
			for j in range(skeletons.shape[0]):
				color = (255*(j%2), 0, 255*((j+1)%2))
				for k in range(skeletons.shape[2]):
					cv2.circle(img,(skeletons[j][i][k][0],skeletons[j][i][k][1]), 5, color, -1)
					
			
			cv.putText(img,label_dict[label],(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
			out.write(img)
		
		out.release()
		

    def accuracy(self, outputs, labels):
        predictions = np.argmax(outputs.detach().numpy(), axis=1)
        return np.sum(predictions == labels.numpy())/len(outputs)