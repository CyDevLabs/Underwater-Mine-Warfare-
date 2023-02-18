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
