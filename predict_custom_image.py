import deepneuralnet as net
import random
import tflearn.datasets.mnist as mnist
import numpy as np
from skimage import io
import sys

model = net.model
path_to_model = 'final-model.tflearn'
path_to_image = sys.argv[1] # Change this to the file path/name of the image file you want to use

model.load(path_to_model)

# Load image (normalized)
x = io.imread(path_to_image).reshape((28, 28, 1)).astype(np.float) / 255

result = model.predict([x])[0] # Predict

prediction=0
probability=result[0]
for i in range(10):
    if probability < result[i]:
        prediction=i
        probability=result[i]

print("Prediction:",prediction , "Probability %:",probability*100)
