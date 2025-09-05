# Medical Diagnosis
## Itroduction 
Computer vision is an interdisciplinary scientific field that deals with how computers can gain a high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. We can use Computer Vision to determine whether a person is affected by pneumonia or not.

Computer Vision (CV) has a lot of applications in medical diagnosis:

•	Dermatology

•	Ophthakmology

•	Histopathology.

X-rays images are critical for the detection of lung cancer, pneumenia ... In this notebook you will learn:

•	Data pre-processing

•	Preprocess images properly for the train, validation and test sets.

•	Set-up a pre-trained neural network to make disease predictions on chest X-rays.
### Pneumonia
Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.
Pneumonia can range in seriousness from mild to life-threatening. It is most serious for infants and young children, people older than age 65, and people with health problems or weakened immune systems.
<img width="601" height="423" alt="image" src="https://github.com/user-attachments/assets/18906c9b-e969-4739-8a9d-647fe82fa928" />
### Pneumonia Detection with Convolutional Neural Networks
Computer Vision can be realized using Convolutional neural networks (CNN) They are neural networks making features extraction over an image before classifying it. 

The feature extraction performed consists of three basic operations:

•	Filter an image for a particular feature (convolution)

•	Detect that feature within the filtered image (using the ReLU activation)

•	Condense the image to enhance the features (maximum pooling)
<img width="868" height="400" alt="image" src="https://github.com/user-attachments/assets/2829605d-33bc-4939-84ca-f5e44044962a" />

Using convolution filters with different dimensions or values results in differents features extracted.

Features are then detected using the reLu activation on each destination pixel.

<img width="422" height="250" alt="image" src="https://github.com/user-attachments/assets/e48e49ff-1565-4885-adbc-98dc851496c9" />

Features are the enhanced with MaxPool layers

<img width="780" height="250" alt="image" src="https://github.com/user-attachments/assets/5fd6efe8-3715-460b-8739-73c8de421729" />

The stride parameters determines the distance between each filters. The padding one determines if we ignore the borderline pixels or not (adding zeros helps the neural network to get information on the border)

<img width="537" height="423" alt="image" src="https://github.com/user-attachments/assets/56458156-3ab2-44ec-844f-3ccce6158229" />

The outputs are then concatened in Dense layers

<img width="457" height="348" alt="image" src="https://github.com/user-attachments/assets/d146e1a6-6382-4bc8-8cfe-217e03ea1e13" />

By using a sigmoid activation, the neural network determines which class the image belongs to

<img width="529" height="431" alt="image" src="https://github.com/user-attachments/assets/c9983b0e-bc1c-4191-aa54-b0d90c63f712" />

## Image Preprocessing
Before training, we'll first modify your images to be better suited for training a convolutional neural network. For this task we'll use the Keras ImageDataGenerator function to perform data preprocessing and data augmentation.
This class also provides support for basic data augmentation such as random horizontal flipping of images. We also use the generator to transform the values in each batch so that their mean is 0 and their standard deviation is 1 (this will faciliate model training by standardizing the input distribution). The generator also converts our single channel X-ray images (gray-scale) to a three-channel format by repeating the values in the image across all channels (we will want this because the pre-trained model that we'll use requires three-channel inputs).
### Generator for validation and test sets
Separate generator is built since the same generator built for training data can’t be used. It normalizes each image per batch, meaning thatit uses batch statistics. We should not do this with the test and validation data, since in a real life scenario we don't process incoming images a batch at a time (we process one image at a time). Knowing the average per batch of test data would effectively give our model an advantage (The model should not have any information about the test data). What we need to do is to normalize incomming test data using the statistics computed from the training set.
## CNN model
Impact of imbalance data on loss function
Loss Function:
<img width="639" height="51" alt="image" src="https://github.com/user-attachments/assets/0dfc7a8a-8a29-4e19-8546-6ae8ad9eadee" />
We can rewrite the the overall average cross-entropy loss over the entire training set D of size N as follows:
<img width="836" height="86" alt="image" src="https://github.com/user-attachments/assets/dbe03bb3-c914-442d-b4ab-07d9046a45a5" />
When we have an imbalance data, using a normal loss function will result a model that bias toward the dominating class. One solution is to use a weighted loss function. Using weighted loss function will balance the contribution in the loss function.
<img width="663" height="70" alt="image" src="https://github.com/user-attachments/assets/465b05ad-41e8-4927-aa50-27d0025fccdc" />
## Transfer Learning
### DenseNet
Densenet is a convolutional network where each layer is connected to all other layers that are deeper in the network:
•	The first layer is connected to the 2nd, 3rd, 4th etc.
•	The second layer is conected to the 3rd, 4th, 5th etc.
<img width="761" height="487" alt="image" src="https://github.com/user-attachments/assets/0ece4bc3-1e11-4776-9555-85af77892b3c" />
### VGG16
Presented in 2014, VGG16 has a very simple and classical architecture, with blocks of 2 or 3 convolutional layers followed by a pooling layer, plus a final dense network composed of 2 hidden layers (of 4096 nodes each) and one output layer (of 1000 nodes). Only 3x3 filters are used.
<img width="806" height="412" alt="image" src="https://github.com/user-attachments/assets/395cd822-5333-4717-ae54-8d46d532a9ba" />
### ResNet
Residual Network, is a type of deep neural network architecture that uses skip connections to enable the training of much deeper networks than previously possible. By allowing information to bypass layers, ResNets solve the vanishing gradient problem, improving performance in deep learning tasks like image recognition, and have led to variants such as the 50-layer ResNet (ResNet50).

<img width="461" height="276" alt="image" src="https://github.com/user-attachments/assets/a66d6271-dcad-454e-91da-55a76d62009b" />
### InceptionNet
InceptionNet uses Inception Modules to provide more efficient computation and deeper networks by reducing the dimensionality of the network with stacked 1X1 convolutions. The modules were created to address various difficulties, including computational expense and overfitting.
<img width="859" height="501" alt="image" src="https://github.com/user-attachments/assets/b31b1bad-39ea-45ba-b478-39062221cd3b" />
