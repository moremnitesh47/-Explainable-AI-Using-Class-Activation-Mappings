# Explainable AI using class activation mappings in Pytorch
Authors- kashif Riyaz, Nitesh morem.

 Ostbayerische Technische Hochschule Amberg-Weiden.
 
 Department of Electrical Engineering, Media and Computer Science.

# Introduction

 The deep neural networks have been the hot topic of this century, revolutionizing
 various fields with their remarkable performance. However, the nature of these
 neural networks, ”black-box” often highlights some concerns about transparency
 and interpretability of these networks, hindering trust and broader adoption in
 critical applications. This project aims to improve the interpretability of deep
 learning models through Class Activation Mapping and techniques associated
 with Gradient-Weighted Class Activation Mapping. As AI models are deployed
 everywhere around the globe especially in the high-stake institutions such as
 Healthcare, automotive industries, Finance and Banking, it becomes crucial to
 understand the level of trust these AI models can provide in real life scenarios.
 This research will bridge the existing gap between understanding intricate CNN
 decision-making, trusting these models, and using them since there is currently
 a lack of standard methods for their interpretation. In order to understand and
 address this, our project unveils the inner workings of AI models and explains
 the decision-making process by visual explanations. By this we aim not just for
 building Trust and confidence in AI systems but also ensure that they are used
 ethically and safely for applications through this project.
 The basis of this project is to understand the behavior of deep neural net
works through two important methods: visualization of layer-specific embed
dings and different class activation mappings. Layer-wise embeddings bring
 out the learned representations from the layers of a network before the output,
 which has clear insight with respect to how the model processes information in
 the background. On the other hand, class activation mappings highlight relevant
 regions in input data for specific class predictions, giving a visual explanation of
 the model’s decisions. The goals of this research involve class implementation of
 pre-trained models which are trained on Imagenet dataset, testing them on di
verse data, and setting up a extendable standardized framework for pre-trained
 models in Torch.nn.


![image](https://github.com/user-attachments/assets/248114b7-60c4-4c83-9690-f9d1fd6eb3e2)
Source:https://github.com/jacobgil/pytorch-grad-cam



 Grad-CAM overview: Given an image and a class of interest (e.g., ‘tiger
 cat’ or any other type of differentiable output) as input, we forward propagate
 the image through the CNN part of the model and then through task-specific
 computations to obtain a raw score for the category. The gradients are set
 to zero for all classes except the desired class (tiger cat), which is set to 1.
 This signal is then back propagated to the rectified convolutional feature maps
 of interest, which we combine to compute the coarse Grad-CAM localization
 (blue heatmap) which represents where the model has to look to make the
 particular decision. Finally, we pointwise multiply the heatmap with guided
 backpropagation to get Guided Grad-CAM visualizations which are both high
resolution and concept-specific.


 # CAM visualization techniques:

 This section uses visualization techniques such as Grad-CAM, Grad-CAM++,
 Eigen-CAM, and XGrad-CAM. When applied each technique takes the model
 and layer number as input and generates a heat map which highlights the regions
 in the image that contribute most to the target class prediction. These CAM
 techniques have been imported from jacob gill’s github repository and his github
 Tech Blog called “Advanced AI Explainability with pytorch-gradcam ”

  In this section the heat maps from the each CAM technique are masked on the
 image as shown in Figure
 ![image](https://github.com/user-attachments/assets/d2c72ba3-b345-4c20-bd3c-879cf422ee49)


 # Results and Discussions
 
 The class activation mapping (CAM) techniques were implemented on various
 pre-trained models, including DenseNet121, ResNet152, and inception v3, each
 with a different number of convolutional layers. For analysis, the mappings
 were examined with different layer numbers: the first, middle, and last convolu
tional layers of each model. The CAM techniques were applied to a diverse set
 of images, including both single-label and multi-label images, to evaluate the
 consistency and robustness of the explanations provided by these models.
 
 
 In this report we have only shown the heatmaps of only three models which
 are as follows,
 • Densnet121 with 120th convolutional layers.
 • Resnet152 with 155th convolutional layers.
 • Inception v3 with 96th convolutional layers.

  # --FOR RESNET152
  <li>Heat map for first Layer resnet152.</li>
  ![image](https://github.com/user-attachments/assets/0a6d9a28-cef4-4d83-8b81-f05049404b6e)

  <li>Heat map for middle layer resnet152.</li>
  ![image](https://github.com/user-attachments/assets/be6f8272-19ca-4e73-8d88-355f548d440b)

  <li>Heat map for Last layer resnet152.</li>
  ![image](https://github.com/user-attachments/assets/87ae7213-1694-4a14-936d-c1fce2a37e7e)


# --For DENSNET121

  <li>Heat map for first Layer Densnet121.</li>
  ![image](https://github.com/user-attachments/assets/1e492305-a3d9-4cff-849d-7aa45b29f2f9)


  <li>Heat map for middle layer Densnet121.</li>
   ![image](https://github.com/user-attachments/assets/c95f3b22-825e-4b00-9e78-802cc55aff5d)


  <li>Heat map for Last layer Densnet121.</li>
  ![image](https://github.com/user-attachments/assets/1332cda1-19c1-4c1f-835b-acef9db13f18)

# --For INCEPTION_V3

  <li>Heat map for first Layer inception_v3.</li>
   ![image](https://github.com/user-attachments/assets/a47630c9-9cea-471c-931e-9c01bf0eed78)



  <li>Heat map for middle layer inception_v3.</li>
   ![image](https://github.com/user-attachments/assets/37d8053b-c361-45bb-9428-cedf1d8c7ed4)



  <li>Heat map for Last layer inception_v3.</li>
  ![image](https://github.com/user-attachments/assets/48b91662-4122-4467-a79d-9e5bd954681d)


   In the last convolutional layer, all models perform well with CAM techniques.
 They create heat maps that accurately show features important for classifying
 images. This shows how effective CAM techniques are at revealing these crucial
 features that help the models make accurate predictions.




 

 #References
 [1] R. R. Selvaraju et al., “Grad-cam: Visual explanations from deep networks
 via gradient-based localization,” IEEE International Conference on Com
puter Vision (ICCV), pp. 618–626, 2017.
 [2] J. Gildenblat and contributors, Pytorch library for cam methods, https:
 //github.com/jacobgil/pytorch-grad-cam, 2021.
 [3] A. Chattopadhay et al., “Grad-cam++: Generalized gradient-based visual
 explanations for deep convolutional networks,” IEEE Winter Conference
 on Applications of Computer Vision (WACV), pp. 839–847, 2018.
 [4] A. Muhammad et al., “Eigen-cam: Class activation map using principal
 components,” IEEE Transactions on Neural Networks and Learning Sys
tems, pp. 1–12, 2020.
 [5] R. Fu et al., “Xgrad-cam: Towards accurate and explainable ai,” arXiv
 preprint arXiv:2008.02312, 2020.





 
