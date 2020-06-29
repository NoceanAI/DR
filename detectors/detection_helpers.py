import imutils
import json 


def sliding_window(image, step, ws):
	# slide a window across the image
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# yield the current window
			yield (x, y, image[y:y+ws[1], x:x+ws[0]])


def image_pyramid(image, scale=1.5, minSize=(224, 224)):
	# yield the original image
	yield image
	# keep looping over the image pyramid
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image


def decode_predictions(preds, top=1, class_list_path='index.json'):
	if len(preds.shape) != 2 or preds.shape[1] != 5: # your classes number
		raise ValueError('`decode_predictions` expects'
		                 'a batch of predictions '
		                 '(i.e. a 2D array of shape (samples, 5)). '
		                 'Found array with shape: ' + str(preds.shape))
	index_list = json.load(open(class_list_path))
	results = []
	for pred in preds:
		top_indices = pred.argsort()[-top:][::-1]
		result = [tuple(index_list[str(i)]) + (pred[i],) for i in top_indices]
		result.sort(key=lambda x: x[2], reverse=True)
		results.append(result)
	return results