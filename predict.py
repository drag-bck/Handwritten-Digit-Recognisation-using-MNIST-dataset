import deepneuralnet as net
import random
import tflearn.datasets.mnist as mnist
from skimage import io
import numpy

model = net.model
path_to_model = 'final-model.tflearn'

_, _, testX, _ = mnist.load_data(one_hot=True)
model.load(path_to_model)

# Randomly take an image from the test set
rand_index = random.randint(0, len(testX) - 1)
x = testX[rand_index].reshape((28, 28, 1))

result = model.predict([x])[0] # Predict

prediction=0
probability=result[0]
for i in range(10):
    if probability < result[i]:
        prediction=i
        probability=result[i]


#prediction = result.index(max(result)) # The index represents the number predicted in this case
print("Prediction:",prediction , "Probability %:",probability*100)

io.imsave('testimage.jpg', x.reshape(28, 28)) # This shows the image in the computer for you to see