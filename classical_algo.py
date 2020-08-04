import numpy as np 
import argparse 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from refine.preprocessing import simplepreprocessor
from refine.preprocessing import imagetoarraypreprocessor
from refine.datasets import simpledatasetloader
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="knn",
help="name of model")
args = vars(ap.parse_args())




def extract_color_stats(image):
	# split the input image into its respective RGB color channels
	# and then create a feature vector with 6 values: the mean and
	# standard deviation for each of the 3 channels, respectively
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B)]
	# return our set of features
	return features



Models = {
	"knn": KNeighborsClassifier(),
	"logit": LogisticRegression(solver = "lbfgs", multi_class = "auto"),
	"SVC": SVC(kernel = "linear")
}



print("loading images...")
# grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print("extracting image features...")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []
# loop over our input images
for imagePath in imagePaths:
	# load the input image from disk, compute color channel
	# statistics, and then update our data list
	image = Image.open(imagePath)
	features = extract_color_stats(image)
	data.append(features)
	# extract the class label from the file path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)


#print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))


le = LabelEncoder()
labels = le.fit_transform(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels,
test_size=0.25, random_state=42)



if args["model"] not in Models.keys():
	raise AssertionError("model name not valid")


model = Models[args["model"]]

model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX),
target_names=le.classes_))