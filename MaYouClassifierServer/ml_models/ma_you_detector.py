import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os



class MaYouDetector:
	def __init__(self, image_size):
		self.model = None
		self.image_size = image_size
		self.data_augmentation = keras.Sequential(
		    [
		        layers.RandomFlip("horizontal"),
		        layers.RandomRotation(0.1),
		    ]
		)

		self.Xception = keras.applications.Xception(
		    weights='imagenet',  
		    input_shape=(image_size),
		    include_top=False
		)

		self.Xception.trainable = False

		self.Ma_You_classify_model_save_path = os.path.join(os.getcwd(), "MaYouClassifierServer/ml_models/Ma_You_detector_weights")
	def make_model(self, num_classes=2):
		inputs = keras.Input(shape=self.image_size)
		# Image augmentation block
		x = self.data_augmentation(inputs)

		# Entry block
		x = layers.Rescaling(1.0 / 255)(x)

		# Go through pretrained Xception model
		x = self.Xception(x, training=False)
		# Convert features of shape `base_model.output_shape[1:]` to vectors
		x = keras.layers.GlobalAveragePooling2D()(x)

		if num_classes == 2:
		    activation = "sigmoid"
		    units = 1
		else:
		    activation = "softmax"
		    units = num_classes
		x = layers.Dropout(0.5)(x)
		outputs = layers.Dense(units, activation=activation)(x)
		self.model = keras.Model(inputs, outputs)
		self.model.load_weights(filepath=self.Ma_You_classify_model_save_path)