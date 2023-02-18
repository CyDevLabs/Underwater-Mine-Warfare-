# [In Progress] 
# Mini-Research-Project based on Deep Learning applied to Underwater Mine Warfare 

**Google Colab using Tensorflow**

training an autoencoder on a single image may not yield useful results as the model may simply learn to memorize the input image without extracting useful features. It's usually better to use a dataset of multiple images to train the autoencoder.

The batch size is a hyperparameter in machine learning that refers to the number of samples used in one iteration of training a neural network. 

**Results**
**1** TensorFlow was able to find 8 images in training dataset, 2 images in your validation dataset, and 2 images in your test dataset, and that the images belong to 2 different classes.

**DAE_code1**

The Flatten layer takes a tensor of shape (128, 128, 3) as input, and flattens it into a 1D tensor of shape (128 * 128 * 3,). This layer is needed because the subsequent layers are fully connected and require a 1D input.

The Dense layer has 512 neurons with ReLU activation. This layer applies a linear transformation to the input tensor, followed by a Rectified Linear Unit (ReLU) activation function.

The second Dense layer has 1024 neurons with ReLU activation. This layer applies another linear transformation to the output of the previous layer, followed by ReLU activation.

The third Dense layer has 512 neurons with ReLU activation. This layer applies another linear transformation to the output of the previous layer, followed by ReLU activation.

The fourth Dense layer has 128 * 128 * 3 neurons, which is equal to the number of pixels in the input image. This layer applies a linear transformation to the output of the previous layer.

The Reshape layer reshapes the output of the previous layer into a tensor of shape (128, 128, 3), which is the same shape as the input image.

The overall purpose of this model is to learn a function that maps a noisy image to a denoised image. The first layer flattens the input image into a 1D tensor, which is then passed through several fully connected layers with ReLU activations, allowing the model to learn a nonlinear function of the input. The last layer reshapes the output back into an image of the same size as the input.

**DAE**

The code you provided creates a Keras Sequential model that performs image denoising using a convolutional neural network (CNN). The model architecture is different from the previous one:

The first layer is a Conv2D layer with 32 filters of size (3, 3). This layer applies 32 convolutional filters to the input image, with ReLU activation and 'same' padding. The input shape is (126, 126, 3), which is slightly smaller than the input shape in the previous model.

The second layer is a MaxPooling2D layer that performs 2x2 max pooling with 'same' padding, reducing the spatial dimensions of the feature maps by a factor of 2.

The third layer is a Conv2D layer with 64 filters of size (3, 3), followed by ReLU activation and 'same' padding.

The fourth layer is another MaxPooling2D layer, with the same parameters as the previous one.

The fifth layer is a Conv2D layer with 128 filters of size (3, 3), followed by ReLU activation and 'same' padding.

The sixth layer is an UpSampling2D layer that performs 2x2 upsampling of the feature maps.

The seventh layer is a Conv2D layer with 64 filters of size (3, 3), followed by ReLU activation and 'same' padding.

The eighth layer is another UpSampling2D layer, with the same parameters as the previous one.

The ninth layer is a Conv2D layer with 32 filters of size (3, 3), followed by ReLU activation and no padding.

The final layer is a Conv2D layer with 3 filters of size (3, 3), followed by ReLU activation and 'same' padding. The output of this layer is a denoised image with the same spatial dimensions as the input.

The purpose of this model is also to learn a function that maps a noisy image to a denoised image, but it does so using a CNN architecture that is more suited to image processing tasks. The Conv2D and MaxPooling2D layers allow the model to learn local features of the input image, while the UpSampling2D layers increase the spatial resolution of the feature maps. The Conv2D layers at the end of the model produce the final denoised image.

The loss function used in this model is binary cross-entropy, and the optimizer is Adam.

**first code provided does not use a convolutional neural network (CNN), but rather a fully connected neural network (also called a dense network). The Flatten layer at the beginning of the model flattens the 3D image tensor into a 1D vector, and the subsequent Dense layers are fully connected layers that learn a function to map the noisy image to the denoised image.
