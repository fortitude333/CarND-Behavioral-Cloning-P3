import csv
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def generator(samples, batch_size=100):
	num_samples = len(samples)
	# batch_size = int(batch_size / 4)
	while 1:
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				angle = float(batch_sample[3])
				correction = 0.2

				# CENTER
				image = get_image(line[0])
				images.append(image)
				angles.append(angle)

				# Augmenting images by flipping across y-axis
				images.append(cv2.flip(image, 1))
				angles.append(-angle)

				# LEFT
				images.append(get_image(line[1]))
				angles.append(angle + correction)

				# RIGHT
				images.append(get_image(line[2]))
				angles.append(angle - correction)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)

def read_data_from_file(data_path='data/'):
	lines = []
	with open(data_path + 'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	images = []
	angles = []
	for line in lines:
		angle = float(line[3])
		correction = 0.2

		# CENTER
		image = get_image(line[0], data_path)
		images.append(image)
		angles.append(angle)

		# Augmenting images by flipping across y-axis
		images.append(cv2.flip(image, 1))
		angles.append(-angle)

		# LEFT
		images.append(get_image(line[1], data_path))
		angles.append(angle + correction)

		# RIGHT
		images.append(get_image(line[2], data_path))
		angles.append(angle - correction)

	X_train = np.array(images)
	y_train = np.array(angles)

	return (X_train, y_train)

def store_in_pickle(data, filename='data.p'):
	pickle.dump(data, open(filename, 'wb'))
	print("Stored data in {}".format(filename))

def get_image(path, base_path='../data/'):
	# load image and conver to RGB
	filename = path.split('/')[-1]
	path = base_path + 'IMG/' + filename
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image


if __name__ == '__main__':
	# lines = []
	# with open('../data/driving_log.csv') as csvfile:
	# 	reader = csv.reader(csvfile)
	# 	for line in reader:
	# 		lines.append(line)

	# train_samples, validation_samples = train_test_split(lines, test_size=0.2, random_state=51)

	# batch_size = 25
	# train_generator = generator(train_samples, batch_size)
	# validation_generator = generator(validation_samples, batch_size)

	# load data 
	X_train, y_train = read_data_from_file('../data/')
	
	# Defining the model
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5 , input_shape=(160, 320, 3)))
	model.add(Cropping2D(cropping=((70, 25), (0, 0))))
	model.add(Conv2D(24, (5, 5), strides=2, activation='relu'))
	model.add(Conv2D(36, (5, 5), strides=2, activation='relu'))
	model.add(Conv2D(48, (5, 5), strides=2, activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Flatten())
	model.add(Dense(1164))
	model.add(Dropout(0.3))
	model.add(Dense(100))
	model.add(Dropout(0.3))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')
	history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, batch_size=100)
	# history_object = model.fit_generator(train_generator,
	# 									 steps_per_epoch=len(train_samples)/batch_size,
	# 									 validation_data=validation_generator,
	# 									 validation_steps=len(validation_samples)/batch_size,
	# 									 epochs=5)

	print(history_object.history.keys())

	# Plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

	# Save model for use by drive
	model.save('model.h5')