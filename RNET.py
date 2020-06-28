from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K 

# RNET 16
class RNET:
	@staticmethod
	def build(width, height, depth, classes):
		# defining the model with input shape and channel
		# dimension


		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# checking if the keras.json file is using the 
		# configuration of "channels_last"
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1


		model.add(Conv2D(64, (3,3), padding = "same", input_shape = inputShape))
		model.add(Activation("relu"))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
		model.add(Conv2D(256, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(512, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the model we have built
		return model



