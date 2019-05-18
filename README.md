# IEOR242_deep_colorization
IEOR242 project at UC Berkeley: automated colorization of gray scale images using deep convolutional neural nets.

...
Write stuff here
...


The testing part does not include any metrics since there is no real metric that really captures the "accuracy" of the model, since the quality of the colorization is such a subjective notion.

Another CNN model was built for the regression model, using tensorflow 2.0. This model is contained in the notebook "Regression_tensorflow2.ipynb". It requires the last version of tensorflow to run. All the steps (loading data, building the model, training and visualization) are part of the notebook. The visualization consists in generating images from the training set and showing the original image, the gray-scale image and the model output next to each other. In order to visualize results on different images, just relaunch the last cell to see the result on the next image of the image generator.

