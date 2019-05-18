# IEOR242_deep_colorization
IEOR242 project at UC Berkeley: automated colorization of gray scale images using deep convolutional neural nets.

This work is strongly inspired from the paper Colorful Image Colorization (http://richzhang.github.io/colorization/)

## How to use the code:
### 1) Setup
You need a GPU in order for the training to be tractable, especially for higher resolutions like (256, 256).
You also need to download either cifar-10 or a database of images like imagenet. By default, cifar-10 is assumed to be in the folder 'cifar-10-batches-py' at the root of the project and imagenet pictures are assumed to be in the folder 'val'.

To download cifar-10, run:
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
```

To download the part of imagenet we used, run:
```
wget http://www.image-net.org/challenges/LSVRC/2010/download/non-pub/ILSVRC2010_images_val.tar
tar -xvf ILSVRC2010_images_val.tar
```

### 2) Training
You can train the network with using the script train.py:
```
python train.py --model (paper(default)/small/tiny)
		--data (imagenet(default)/cifar10)
		--n_data (number of images in the dataset, default=50000, this is to deduce how many steps are needed for 1 epoch)
		--batch_size (default=40)
		--resolution (default=104, set it to 32 for cifar-10)
		--n_bins (default=13, number of 1 dimensional bins in the discretization for classification, the total number of bins/classes will be n_bins^2)
		--regression (flag to use the regression versions of the networks)
		--logdir (default='tensorboard', logs directory for tensorboard)
```
Example to train on imagenet with resolution (104, 104):
```
python train.py --model paper --data imagenet --n_data 50000 --batch_size 40 --resolution 104 --n_bins 13
```
### 3) Results
To visualize results, you can use the notebook model_experimentation.ipynb and add a code block similar to the ones already written.

## Code Structure
train.py: This is the main file, it launches the training with the given options.
utils.py: It contains pre-processing function and python generators to go through the data.
models.py: It contains the three different models that we tried with tensorflow 1.X
tests/test_utils.py: It contains unit tests for the pre-processing, make sure these runs without problems before trying to debug the models.
cifar_10_exp__tf_20.ipynb: First results with cifar-10 and tensorflow 2.0
model_experimentation.ipynb: Loading weights, visualizing and saving results. /!\ The trained weights are NOT provided because the models are too big and also because they are not satisfactory enough to our standards /!\
pre_process_visualization.ipynb: To visualize the pre-processing.

The testing part does not include any metrics since there is no real metric that really captures the "accuracy" of the model, since the quality of the colorization is such a subjective notion.

## Using tensorflow 2.0
Another CNN model was built for the regression model, using tensorflow 2.0. This model is contained in the notebook "Regression_tensorflow2.ipynb". It requires the last version of tensorflow to run. All the steps (loading data, building the model, training and visualization) are part of the notebook. The visualization consists in generating images from the training set and showing the original image, the gray-scale image and the model output next to each other. In order to visualize results on different images, just relaunch the last cell to see the result on the next image of the image generator.

