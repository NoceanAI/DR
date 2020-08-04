import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt 
#matplotlib.use("Agg")

# importing packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from refine.preprocessing import simplepreprocessor
from refine.preprocessing import imagetoarraypreprocessor
from refine.datasets import simpledatasetloader
from RNET import RNET
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from sklearn.model_selection import train_test_split
from imutils import paths
import tensorflow as tf 
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


lr_init = 0.0001

ap = argparse.ArgumentParser()
#ap.add_argument("-o", "--output", required = True, help = "path to the output pllot")
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
help="path to output model")
args = vars(ap.parse_args())

classLabels = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]


print("Loading in image data")
imagePaths = list(paths.list_images(args["dataset"]))
#idxs = np.random.randint(0, len(imagePaths), size=(10,))
#imagePaths = imagePaths[idxs]

sp = simplepreprocessor.SimplePreprocessor(32, 32)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

sdl = simpledatasetloader.SimpleDatasetLoader([sp, iap])

(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype("float") / 255.0


trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.20, random_state = 0)


trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
#print(testX.shape)

opt = Adam(lr = lr_init, decay = lr_init/100)
model = RNET.build(width = 32, height = 32, depth = 3, classes = 5)
print("compiling................ Please Wait...")
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
#plot_model(model, to_file = "RNET.png")
print(model.summary())

print("[CAUTION] Training about to occur, do not interrupt")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=200, verbose=1)
#model.fit_generator(trainGen, batch_size=32, epochs=100, verbose=1, shuffle = True)

print("Serializing Network")
model.save(args["model"])

# evaluating the network
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 200), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 200), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 200), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 200), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()