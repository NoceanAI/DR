# import the necessary packages
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		image = cv2.GaussianBlur(image,(5,5),0)
		# denoising of image saving it into dst image 
		#image = cv2.blur(image, (10,10)) 
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)
