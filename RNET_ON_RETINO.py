import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt 
#matplotlib.use("Agg")

# importing packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from refine.preprocessing import simplepreprocessor
from refine.preprocessing import imagetoarraypreprocessor
from refine.datasets import simpledatasetloader
from RNET import RNET
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from imutils import paths



ap = argparse.ArgumentParser()
#ap.add_argument("-o", "--output", required = True, help = "path to the output pllot")
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-a", "--augment", type=int, default=-1, help="whether or not 'on the fly' data augmentation should be used")
ap.add_argument("-m", "--model", required=True,
help="path to output model")
args = vars(ap.parse_args())

classLabels = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]


print("Loading in image data")
imagePaths = list(paths.list_images(args["dataset"]))
#idxs = np.random.randint(0, len(imagePaths), size=(10,))
#imagePaths = imagePaths[idxs]

sp = simplepreprocessor.SimplePreprocessor(64, 64)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

sdl = simpledatasetloader.SimpleDatasetLoader([sp, iap])

(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype("float") / 255.0


trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.25, random_state = 42)


trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
#print(testX.shape)

aug = ImageDataGenerator()

if args["augment"] > 0:
	print("performing 'on the fly' data augmentation")
	aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

print("compiling................ Please Wait...")
opt = SGD(lr = 0.005, momentum = 0.9, decay = 0.005/100)
model = RNET.build(width = 64, height = 64, depth = 3, classes = 5)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
print(model.summary())

print("[CAUTION] Training about to occur, do not interrupt")
H = model.fit(
	x = aug.flow(trainX, trainY, batch_size=32), 
	validation_data=(testX, testY), 
	epochs=100, verbose=1)

print("Serializing Network")
model.save(args["model"])

# evaluating the network
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()