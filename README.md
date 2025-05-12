# Power system oscillation location

Abstract: With the rapid development of large-scale power systems, forced oscillations (FOs) are becoming more and more important, and accurate and rapid location of interference sources has become the key to ensure the stable operation of the system.
Based on the WECC 179-node system, a multi-level time-frequency feature extraction method combining PCA and Smooth Pseudo-Wigner-City Distribution (SPWVD) is proposed.Transform the vibration localization problem at the system and area level into an image recognition task.Using a pre-trained VGG16 model, a two-stage deep transmission learning framework (DTL) was designed to perform vibration source localization at the system and areaal levels.Through the automatic processing of batch destruction simulation data and time-frequency graphs, combined with the deep learning model, the high precision, good robustness and training efficiency of the proposed method are ensured, and the high precision, good durability and training efficiency of the proposed method are ensured.While ensuring data protection and reducing communication pressure, the proposed method is significantly better than the traditional positioning method, and has significant improvement compared with other models.
	Introduce
1.1Background
The forced oscillation of the power system is a system oscillation phenomenon caused by external periodic disturbances, which is widely present in hydropower turbines, wind turbines and load fluctuations. The persistence of these conditions can lead to abnormal operation of equipment and even system instability, and in severe cases, it may lead to large-scale power outages [1-3]. Therefore, it is of great significance to accurately locate the source of oscillation disturbance to ensure the safety of the power grid and improve the stability of operation. 
Traditional localization methods, such as the Energy Flow Method (DEF) [4], the graph theory-based method: The topology of the distribution network is analyzed, the relationship between system nodes and distributed generators is integrated into the network description matrix, and the improved matrix algorithm is used to locate the fault segments[5], and A novel single-ended localization method is based on transient information fusion matrix and waveform similarity technology [6]., have problems such as strong dependence on system models, high computational complexity, and poor adaptability to complex working conditions, although they can locate the oscillation source to a certain extent. With the widespread deployment of synchrophasor measurement units (PMUs) and wide-area measurement systems (WAMS), the massive amount of data collected in real time opens up new possibilities for data-driven oscillatory positioning.
Deep learning (DL), as a powerful tool for automatic feature extraction and nonlinear modeling, has achieved great success in the fields of image recognition and speech processing [7-9], and has also been introduced into power system oscillation localization in recent years, showing excellent performance [10-12]. However, directly using raw time series data to train deep models has problems such as high data dimension, high noise, and long training time. To this end, Shuang Feng et al. proposed a time-frequency graph that combines principal component analysis (PCA) and smoothed pseudo-Wigner-Ville distribution (SPWVD) to generate oscillating signals, and uses a two-stage deep transfer learning framework to realize system-level and area-level oscillation source localization, which significantly improves the accuracy and training efficiency.
	Based on the above theories, this paper designs and implements a more complete training strategy for disturbance simulation data generation, batch automatic processing of time-frequency graphs, and transfer learning, which further improves the positioning performance and model generalization ability, and provides an efficient and practical technical solution for oscillation localization of large-scale power systems.
 
Fig1.Bus_model 
1.2Power system dynamics model
	The dynamic behavior of power systems under forced oscillatory perturbations is usually described by nonlinear differential algebraic equations (DAEs) [14]:
{█((x(t)=Ax(t)+Bμ(t)) ̇@y(t)=Cx(t)+Dμ(t)@x(0)=x_0 )┤
where x∈R^nstate variable vectors, such as generator rotor angle and velocity; μ(t) is an external perturbation input and is often expressed as a sinusoidal signal:
μ_k (t)=∆Fsin(ωt)
Corresponds to the perturbation amplitude ΔF and frequency ω. Perturbations cause the measurement signal to continue to oscillate, and the multi-node timing data collected by the PMU contains the position information of the perturbation source [13].
1.3 The oscillation source locates the target
	The goal is to identify the specific areas and nodes of disturbance injection based on the multi-node oscillation signals collected by the PMU, so as to achieve fast and accurate location of the forced oscillation source. This problem can be translated into an analytical sampling matrix Y:
■(Y&=)[y_1,y_2,…,y_n]ϵR^(n×M)
M is the number of sampling points, and n is the number of measurement nodes.
Traditional methods are difficult to process high-dimensional nonlinear data, and deep learning methods are adopted because of their ability to automatically extract complex features
1.4 Experimental environment
Software and hardware environments :MATLAB R2019b,python 3.8, TensorFlow 2.6, Keras deep learning framework, used for model training and evaluation, and the training hardware is NVIDIA GTX 3080 GPU.
	Data processing
2.1 Data Preparation
First of all, the initial variables and algebraic variables of each equipment (generator, excitation system, load, etc.) in the power system should be set to ensure that they start from a reasonable starting point, and then the gcall function should be used to calculate the algebraic equations of the system, such as node power balance, voltage amplitude and phase angle, etc., to ensure that the steady-state conditions of the power grid are satisfied, and the algebraic equations reflect the static constraints of the system. It is the basis for every step in the dynamic simulation. Then, the differential equations of the dynamic states of the generator rotor angle, velocity, and excitation voltage are calculated through the fcall function to simulate the dynamic behavior of the power system. Then, the Jacobian matrices corresponding to the state variables and algebraic variables are calculated to achieve the fast convergence of the Newton iteration and other numerical solution algorithms, and the island detection function gisland is called in each calculation to ensure that the system topology is correct and avoid the isolated nodes affecting the simulation accuracy and save the data.
2.2Data preprocessing
First, the multi-node time series data containing the active power (p), the reactive power(q), and the voltage (v) are read. For different node groups or areaal nodes in the system, PCA is used to reduce the dimensionality of high-dimensional time series data, extract the most important change features, and reduce the impact of data redundancy and noise. The first principal component of PCA was selected as the representative signal to reflect the core dynamic characteristics of the area or node group. Subsequently, the time-frequency analysis method of these principal component signals, the Wigner-Ville distribution (WVD), was applied to calculate the time-frequency energy distribution map, and the variation law of the signal in time and frequency can be seen. Finally, the obtained time-frequency maps are stitched and combined to form a system-level or area-level two-dimensional image, which is used as the input of the subsequent deep learning model. This processing process not only retains the time-varying frequency characteristics of the disturbance signal, but also effectively reduces the dimensionality, which is convenient for the model to capture the spatial and dynamic characteristics of the disturbance source and improve the classification performance. 

 
Fig2. Global_Time-frequency energy distribution diagram
The figure on the left shows a relatively stable and continuous frequency component, and the energy distribution is concentrated and obvious, showing the continuous oscillation characteristics of the perturbation event in a specific frequency range. Capturing a single source of disturbance or a relatively simple oscillation pattern in a power system allows the model to identify obvious and persistent dynamic behaviors. The figure on the right shows a richer frequency component and multi-time energy distribution, reflecting a complex multi-modal oscillation or multi-stage perturbation process. The frequency components show multi-peak and multi-fluctuation characteristics on the time axis, reflecting the complex dynamic response of multi-node and multi-disturbance source interaction in the power system. 

Table1.Areaal time-frequency diagram 
 
(1)	 
(2)
 
(3)	 
(4)

The time-frequency energy distribution of the principal component signal in the four areas is shown, and the yellow area represents the high energy, which shows the main frequency components of the disturbance signal in each area and their changes with time. The low-frequency energy is weak, and the energy distribution is uniform, indicating that the disturbance is relatively stable and there is no obvious instantaneous impact. The subtle differences in the images of each area reveal the differences in the perturbation response of different areas of the system.
The time-frequency images are then divided into system level and zone level. System-level time-frequency images are used for the first stage of model training, and area-level time-frequency images are used for the second stage of fine-tuning. Uniform image size (e.g., 224×224) meets VGG16 input requirements. Normalize pixel values (normalized to 0~1 or mean variance normalized) to improve training stability.
3.Model
In this paper, the VGG16 convolutional neural network is used, and then a two-stage transfer learning is carried out, and the first stage of the process is to train the basic model based on system-level time-frequency data to capture the overall oscillation characteristics. The second stage is to freeze some layers of the pre-trained model and fine-tune the later layers with area-level data to improve the accuracy of local oscillation source positioning.
3.1Model introduce
VGG16 is a widely used deep convolutional neural network architecture proposed by Simonyan and Zisserman in 2014 [15]. The network consists of 16 weight layers, including 13 convolutional layers and 3 fully connected layers. It employs small 3×3 convolutional filters stacked consecutively, which allows increased depth while keeping computational efficiency. VGG16 demonstrated excellent performance on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014, and its simple and uniform architecture facilitates transfer learning and feature extraction in various computer vision tasks.




3.2Learning	process 
Fig3. Learning process
The first stage is basic model training, which uses system-level data to train the pre-trained model to capture the oscillation characteristics of the power system as a whole. In this stage, large-scale system-level samples are used to help the model learn the general oscillation patterns and feature expressions. In the second stage, some of the parameters of the pre-trained layer are frozen on the basic model trained in the first stage, and only the parameters of the network of the later layer are fine-tuned. By using areaal-level data, the model's ability to identify local oscillation sources is enhanced, so as to improve the accuracy of areaal positioning of oscillation sources.
3.3Preparation
3.3.1 Load the pre-trained model and freeze the convolutional layers
	Load the previously trained system-level model and use it as a feature extractor. By setting all layers of the model to be untrainable (frozen), the weights of its convolutional layers can be avoided from being modified during the current training. This can take advantage of the rich features that the pre-trained model has learned, reduce training time, and prevent overfitting.
 
3.3.2 Building the top-level classifier
In order to adapt to the current classification task, a new set of layers is added behind the frozen base model, firstly, the multi-dimensional convolutional features are flattened by the Flatten layer, then the higher-level features are extracted by the fully connected layer, the dropout layer is added to prevent overfitting, and finally the probability of each category is output by the softmax output layer to achieve multi-classification.
 
3.3.3Compile model
In the compilation stage of the model, the stochastic gradient descent (SGD) optimizer is used to set a small learning rate to ensure the stability of the training, and cross-entropy is used as the loss function, which is suitable for multi-classification tasks, and the accuracy index is monitored during the training process to evaluate the model performance.
 
3.3.4 Set EarlyStopping callback
In order to avoid model overfitting and wasting training time, the EarlyStopping callback function is used to monitor the accuracy on the validation set. If there is no improvement in the accuracy of the verification after 5 consecutive epochs, the training will end early and automatically revert to the best performing weight.
 
3.4 Training model
Use the model's FIT method for training, pass in training and validation data, set the maximum number of training rounds to 30, and enable the early stop mechanism. During training, only the weights of the newly added classification layer at the top level will be updated, and the base model will remain frozen, ensuring faster training and better generalization.
 


4.Visualization
4.1 Accuracy and loss curve
4.1.1 Accuracy curve
 
It can be seen from the figure that both the training accuracy and verification accuracy curves show a stable upward trend, with a faster increase in the initial stage, indicating that the model has learned effective features from the data quickly. As the number of training rounds increases, the training accuracy gradually approaches 90%, and the verification accuracy also increases synchronously and remains at a similar level, indicating that the model has good generalization ability and no obvious overfitting phenomenon. The two curves are relatively stable in the later stage of training, and the verification accuracy is occasionally slightly higher than the training accuracy, which may be due to slight fluctuations caused by the distribution characteristics of the verification set samples or the randomness of the data.
4.1.2 loss curve
 
From the graph, we can see that both the training loss and the validation loss are high in the initial stage, but they decrease rapidly with the increase of training rounds. The overall trend is stable and decreases synchronously, indicating that the model has effectively learned the key features in the data. The training loss is slightly higher than the validation loss, indicating that the model is not overfitting and performs well on the validation set. As the training progresses, the loss tends to be stable and the model gradually converges.

 
This loss curve shows the change in loss under a longer training cycle. Both the training loss and the validation loss continue to decrease, proving that the model training is stable, and the validation loss has not rebounded significantly, further verifying the generalization ability of the model. The loss fluctuations in the middle and late stages are small, indicating that the model has reached a better state in the convergence process.
In general, these two loss curves show that the training process of the deep learning model based on time-frequency diagram is stable, converges well, and has strong generalization ability. It can effectively avoid overfitting and ensure the accuracy and reliability of oscillation source classification.
4.2 Confusion Matrix
 
Fig.4confusion matrix of system 
This system-level confusion matrix shows that the test samples are mainly distributed in the Label2 category. The model can correctly identify 5838 Label2 samples, accounting for the majority of the total samples in this category. At the same time, there are a certain number of misclassifications, with 480 samples misclassified as Label1, 400 misclassified as Label3, and 592 misclassified as Label4, indicating that the model still has some confusion when distinguishing Label2 from other categories, indicating that the model's recognition of the Label2 category is relatively accurate.
 
Fig5.confusion matrix of area 
The confusion matrix of the regional model shows a more balanced and significantly improved classification performance. The number of correctly identified samples on the diagonal of each category exceeds 4800, and the number of misclassifications is significantly reduced, especially the Label4 category, which has 5508 correctly identified samples, and performs best. This shows that the regional model is more sensitive and accurate in extracting fine-grained oscillation features, and can effectively distinguish oscillation sources in different regions. Misclassifications are mainly distributed between adjacent categories, showing the advantages of regional models in capturing local features. Therefore, the regional model can more precisely locate system oscillations, assist the system-level model in making overall judgments, and realize multi-level and multi-angle oscillation source identification, which helps to improve the accuracy and reliability of power system oscillation diagnosis.
5. Innovation and shortcomings
5.1 Innovation
Firstly, the automatic early stop mechanism in the training process uses the accuracy of the EarlyStopping monitoring validation set to prevent overfitting and save computing resources, which is a practical and effective training strategy, and secondly, the design of multi-stage training first trains the system-level model and then trains the area-level model based on its weight, which reflects the hierarchical refinement training strategy, which is helpful for the model to learn layer by layer and improve the overall recognition accuracy. 
5.2 Inadequacy
Freezing all convolutional layers can limit model performance: Freezing all pretrained layers directly prevents overfitting, but it also limits the model's ability to adapt to new data. Fine-tuning some of the high-level convolutional layers may lead to better generalization and performance improvements. 
The data augmentation method is relatively simple: at present, only simple normalization is carried out, and there is a lack of commonly used data augmentation (such as rotation, translation, zoom, flip, etc.), which may limit the robustness of the model in real scenarios.  
Lack of class balancing strategies for the training and validation sets: If the dataset is unevenly distributed in categories, failure to use strategies such as category weighting or resampling may cause the model to be biased toward the majority of classes. 
6. Optimization and improvement
1. Fine-tune the high-level convolutional layer of the pre-trained model:On the basis of freezing the low-level convolutional layer, part of the high-level convolutional layer is gradually thawed for fine-tuning, so that the model can better adapt to the feature distribution of the target data and improve the recognition accuracy and generalization ability.
2. Introduce rich data augmentation techniques:Combined with a variety of data augmentation methods such as rotation, scaling, translation, flipping, and color perturbation, the diversity of training samples is expanded, and the robustness of the model to different environments and sample variations is enhanced.
3. Combine multimodal data and federated learning:If conditions permit, multi-modal information such as sensor data and time series signals can be combined, or distributed training methods such as federated learning can be used to improve the comprehensive recognition ability and data privacy protection of the model.

7.Reference
[1] Kundur, P. (2007). Power system stability. Power system stability and control, 10(1), 7-1.
[2] Anderson, P. M., & Fouad, A. A. (2008). Power system control and stability. John Wiley & Sons.
[3] Milano F. Power System Modelling and Scripting. Springer, 2010.
[4] Tao, J., Bohan, L., Xue, L., & Guoqing, L. I. (2021). Forced oscillation location in power systems using multiple empirical mode decomposition. Proceedings of the CSEE, 42(22), 8063-8074.
[5] Tao, Z., Long, M., & Bowen, L. (2021). Fault section location of active distribution network based on feeder terminal unit information distortion correction. Power System Technology, 45(10), 3926-3934.
[6] WANG, C., LI, J., XU, Z., HAN, J., & LI, Y. (2022). Research on single-ended fault location of transmission line based on transient information fusion. Journal of Electric Power Science and Technology, 37(2), 62-71.
[7] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. nature, 521(7553), 436-444.
[8] Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). Deep learning (Vol. 1, No. 2). Cambridge: MIT press.
[9] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[10] Talukder, S., Liu, S., Wang, H., & Zheng, G. (2021, October). Low-frequency forced oscillation source location for bulk power systems: A deep learning approach. In 2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (pp. 3499-3504). IEEE.
[11] Youssef, A. M., Abdel-Galil, T. K., El-Saadany, E. F., & Salama, M. M. A. (2004). Disturbance classification utilizing dynamic time warping classifier. IEEE Transactions on Power Delivery, 19(1), 272-278.
[12] Shi, Z., Yao, W., Zeng, L., Wen, J., Fang, J., Ai, X., & Wen, J. (2020). Convolutional neural network-based power system transient stability assessment and instability mode prediction. Applied Energy, 263, 114586.
[13] Feng S, Chen J, Ye Y, et al. "A two-stage deep transfer learning for localisation of forced oscillations disturbance source." Electrical Power and Energy Systems, 2022, 135:107577.
[14] Sauer, P. W., Pai, M. A., & Chow, J. H. (2017). Power system dynamics and stability: with synchrophasor measurement and power system toolbox. John Wiley & Sons.
[15] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in *International Conference on Learning Representations (ICLR)*, 2015

