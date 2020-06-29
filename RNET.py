from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K 


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


		model.add(Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = inputShape))
		model.add(SeparableConv2D(32, (3, 3), activation='relu'))
		model.add(Conv2D(32, (3,3), activation = 'relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
		model.add(SeparableConv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
		model.add(SeparableConv2D(128, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		

		


		model.add(Flatten())
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(classes, activation='softmax'))

		# return the model we have built
		return model



