import keras
from keras.api.datasets import mnist
from keras import layers

#train_images and train_labels form the training set, the data that the model will
#learn from. The model will then be tested on the test set, test_images and test_labels.
# The workflow will be as follows: First, we’ll feed the neural network the training data i.e.
# train_images and train_labels. The network will then learn to associate images and
# labels.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#print(train_images[0])
# print(len(train_labels))
# print(train_labels)

# The core building block of neural networks is the layer. You can think of a layer as a fil
# ter for data: some data goes in, and it comes out in a more useful form.
#  layers extract representations out of the data fed into them.
# our model consists of a sequence of two Dense layers, which are densely con
# nected (also called fully connected) neural layers. The second (and last) layer is a 10-way
# softmax classification layer, which means it will return an array of 10 probability scores
# (summing to 1). Each score will be the probability that the current digit image
# belongs to one of our 10 digit classes.

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# An optimizer — The mechanism through which the model will update itself based
# on the training data it sees, so as to improve its performance.

# A loss function — How the model will be able to measure its performance on the
# training data, and thus how it will be able to steer itself in the right direction.

# Metrics to monitor during training and testing — Here, we’ll only care about accu
# racy (the fraction of the images that were correctly classified).

model.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Preprocess the data by reshaping it into the shape the model
# expects and scaling it so that all values are in the [0, 1] interval. Previously, our train
# ing images were stored in an array of shape (60000, 28, 28) of type uint8 with values
# in the [0, 255] interval. We’ll transform it into a float32 array of shape (60000, 28*28)
# with values between 0 and 1.
train_images = train_images.reshape((60000, 28 * 28))
# print(train_images[0])
train_images = train_images.astype("float32") / 255
# print(train_images[0])
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# To train the model, which in Keras is done via a call to the model’s
# fit() method—we fit the model to its training data.
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(predictions[0])
print( predictions[0].argmax())
print(predictions[0][7])
print(test_labels[0])

# compute average accuracy over the entire test set.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")

# The test-set accuracy turns out to be 97.92% and training accurancy is 98.97%.
# This gap between training accuracy and test accuracy is an
# example of overfitting: the fact that machine learning models tend to perform worse
# on new data than on their training data.