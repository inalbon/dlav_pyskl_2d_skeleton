# 2D Skeleton-based Action Recognition

For this project, we started from the code of the paper Revisiting Skeleton-based Action Recognition (Duan et al., 2022). For each frame in a video, they first use a two-stage pose estimator (detection + pose estimation) for 2D human pose extraction. Then they stack heatmaps of joints or limbs along the temporal dimension and apply pre-processing to the generated 3D heatmap volumes. Finally, they use a 3D-CNN to classify the 3D heatmap volumes. The new architecture they proposed is called PoseConv3d. For further information, here is their Github repository [PYSKL repository](https://github.com/kennymckormick/pyskl.git).

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142995620-21b5536c-8cda-48cd-9cb9-50b70cab7a89.png" width=80%/>
</div>

## Contribution
The method developed in the paper of Duan et al. achieves state-of-the-art results. However, it is computationally heavy as we experienced when training their model in SCITAS. Indeed, since they use heatmaps, instead of skeleton keypoints, they have an input shape of (batch_size, #joints, #frames, height, width).

Our idea is to get rid of the heatmap to optimize the computation costs. Instead we use the 2D skeleton data as they are without any modifications to do the classification. This gives an input shape of (batch_size, #persons, #joints, #frames, #dimensions), #dimensions is two for 2D keypoints and three for 3D keypoints. We use nearly the same architecture and adapt the stride, kernel size, and padding of 3D-conv and avg pool. The network was not learning at all, so we try different input shape, we added normalization of the keypoints coordinate.

We coded a data loader in the file dataset.py to load the 2D-skeleton (NTU60 HRNET). In train.py, we load the data, we instantiate and train our model. In inference.py, we predict the data using our best model. We coded functions to display the skeletons on the original video if provided, otherwise we display the skeletons on a black background. The five more probable predictions as well as the true value are plotted on the video. The two models we implemented are in the file c3d_modified.py. The file model.py contains the functions to train the model, evaluate the performance of the model with different metrics, and to do the prediction.

## Experimental Setup
Here we will present the experiments we conducted and the evaluation metrics we use to quatify the results.
### Experiments
First we run the model of the paper with the heatmaps in SCITAS. We obtained these metrics. 
- Top-1 accuracy: 0.9253
- Top-5 accuracy: 0.9944
- Mean class accuracy: 0.9252

Then we adapted the neural network architecture to feed the network with 2D skeletons keypoints instead of heatmaps. We tried different input shape. We adapted the the stride, kernel size, and padding of 3D-conv and avg pool in consequence.
- (batch_size, #joints, #frames, #persons, #dimensions)
- (batch_size, #persons, #joints, #frames, #dimensions)

By changing the order of the two inputs, we thought that it has an effect on the conv3d, because for the first input shape, the conv3d will be applied on the (#frames, #persons, #dimensions) whereas for the  second one, the conv3d will be applied on the (#frames, #joints, #dimensions). The reason we changed the input shape, is that we thought that it was more relevant to put the #joints between #frames and #dimensions to apply the conv3d (it changes the neighborhood of each value).


### Evaluation metrics
Since we made classification, we evaluate the model performance with top-1 accuracy, top-5 accuracy and mean class accuracy. Top-1 accuracy measures the percentage of correct predictions where the predicted class matches the true class. In other words, it calculates the accuracy of the model's most confident prediction. Top-5 accuracy is another evaluation metric used in classification tasks. It measures the percentage of correct predictions where the true class label is found within the top five predicted classes. In this case, the model's prediction does not need to be the most confident one, as long as the true class appears somewhere in the top five predictions. It allows for some flexibility in the predictions and accounts for situations where the model may not be certain about the exact class label. 

We also used MSE loss, precision and recall, F1 score and confusion matrix to quantify our results.

## Dataset
The dataset used for this project is called NTU-RGB+D and was developed by Shahroudy et al. (2016). It contains more than 56,000 video samples collected from 40 distinct subjects. This dataset contains 60 different action classes including daily, mutual, and health-related actions.

Since our project is about Skeleton-based Action Recognition, we use a pre-processed 2D skeletons dataset provided by PYSKL.
- Original Dataset (Shahroudy et al., 2016): &emsp; [NTU-RGB+D](https://arxiv.org/abs/1604.02808)
- Pre-processed dataset (PYSKL):&emsp;&emsp;[NTU-RGB+D 2D SkeletonS download (pickle file)](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl)

<div id="wrapper" align="center">
<figure>
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="520px">&emsp;
  <p style="font-size:1.2vw;">Skeleton-base Action Recognition Results on NTU-RGB+D (PYSKL)</p>
</figure>
</div>

The content of a pickle file is a dictionary with two fields: `split` and `annotations`

1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
   1. `frame_dir` (str): The identifier of the corresponding video.
   2. `total_frames` (int): The number of frames in this video.
   3. `img_shape` (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
   4. `original_shape` (tuple[int]): Same as `img_shape`.
   5. `label` (int): The action label.
   6. `keypoint` (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
   7. `keypoint_score` (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.



## Results
When plotting the top1-accuracy and the loss, it is very clear that the model does not learn at all. Because the accuracy is 0.016 and stays constant along the epochs, the loss should be decreasing along the epochs. However, the loss stays overall constants. This is not encouraging at all.

<div id="wrapper" align="center">
<figure>
  <img src="https://github.com/inalbon/dlav_pyskl_2d_skeleton/assets/82947825/89489c91-6bc9-4ab8-94cc-c6503d92937b" width="520px">&emsp;
  <img src="https://github.com/inalbon/dlav_pyskl_2d_skeleton/assets/82947825/c4f02b5e-ac55-44b5-8ef0-efd2cd03b398" width="350px">&emsp;
  <p style="font-size:1.2vw;">Left: Accuracy and MSE loss of our model;&emsp;Right: Confusion Matrix</p>
</figure>
</div>

Based on these bad results, we decided to check the model sanity by training it with a subset of 100 samples. This is supposed to lead to an accuracy of one due to overfitting. However, as shown in the figure below, the accuracy is far from one. This is a proof that the model is not learning with the data which fed in.

<div id="wrapper" align="center">
<figure>
  <img src="https://github.com/inalbon/dlav_pyskl_2d_skeleton/assets/82947825/89489c91-6bc9-4ab8-94cc-c6503d92937b" width="520px">&emsp;
  <p style="font-size:1.2vw;">Accuracy and MSE loss of the model for 100 samples (sanity check not passed)</p>
</figure>
</div>

It is not relevant to show the other metrics we obtained with the model since it learns nothing from the data.

## Conclusion
au vu des résultats du sanity check, tester autre chose telq que grayscal images
- ca a pris du temps run le code
- c'était dur d'innover alors que tout les concepts étaient nouveaux de base (3dconv, GCN)
- explorer grayscale heatmaps

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```


## Training & Testing

You can use following commands for training and predicting. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
python submission/train.py
# Prediction
python submission/inference.py
 
```

Link to the weights/checkpoints: 


## Bibliography
Revisiting Skeleton-based Action Recognition, (Duan et al, 2022)

