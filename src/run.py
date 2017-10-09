from os import path
import util
import numpy as np
import argparse
from skimage.io import imread
from util import *
import csv
import keras
from keras.models import load_model
import keras.backend as K 
import pdb

import sys
sys.path.append("../model")
import predict



def mean_L1_distance(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def min_L1_distance(y_true, y_pred):
    return K.min(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def max_L1_distance(y_true, y_pred):
    return K.max(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def std_L1_distance(y_true, y_pred):
    return K.std(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

keras.metrics.mean_L1_distance = mean_L1_distance
keras.metrics.min_L1_distance = min_L1_distance
keras.metrics.max_L1_distance = max_L1_distance
keras.metrics.std_L1_distance = std_L1_distance


def categorical_crossentropy_mean_squared_error_1(y_true, y_pred):
    year_pred = K.cast(K.argmax(y_pred,axis = -1), K.floatx())
    year_true = K.cast(K.argmax(y_true,axis = -1), K.floatx())

    return  (K.categorical_crossentropy(y_true, y_pred) + K.cast_to_floatx(1.0) * K.square(year_pred - year_true))

def categorical_crossentropy_mean_squared_error_01(y_true, y_pred):
    year_pred = K.cast(K.argmax(y_pred,axis = -1), K.floatx())
    year_true = K.cast(K.argmax(y_true,axis = -1), K.floatx())

    return  (K.categorical_crossentropy(y_true, y_pred) + K.cast_to_floatx(0.1) * K.square(year_pred - year_true))

def categorical_crossentropy_mean_squared_error_001(y_true, y_pred):
    year_pred = K.cast(K.argmax(y_pred,axis = -1), K.floatx())
    year_true = K.cast(K.argmax(y_true,axis = -1), K.floatx())

    return  (K.categorical_crossentropy(y_true, y_pred) + K.cast_to_floatx(0.01) * K.square(year_pred - year_true))



def pure_mean_squared_error(y_true, y_pred):
    year_pred = K.cast(K.argmax(y_pred,axis = -1), 'float32')
    year_true = K.cast(K.argmax(y_true,axis = -1), 'float32')
    return (LAMBDA * K.square(year_pred - year_true))



def categorical_crossentropy_mean_absoulute_error_1(y_true, y_pred):
    year_pred = K.cast(K.argmax(y_pred,axis = -1), 'float32')
    year_true = K.cast(K.argmax(y_true,axis = -1), 'float32')
    return  (K.categorical_crossentropy(y_true, y_pred) + K.cast_to_floatx(1.0) * K.abs(year_pred - year_true))

def categorical_crossentropy_mean_absoulute_error_01(y_true, y_pred):
    year_pred = K.cast(K.argmax(y_pred,axis = -1), 'float32')
    year_true = K.cast(K.argmax(y_true,axis = -1), 'float32')
    return  (K.categorical_crossentropy(y_true, y_pred) + K.cast_to_floatx(0.1) * K.abs(year_pred - year_true))

def categorical_crossentropy_mean_absoulute_error_001(y_true, y_pred):
    year_pred = K.cast(K.argmax(y_pred,axis = -1), 'float32')
    year_true = K.cast(K.argmax(y_true,axis = -1), 'float32')
    return  (K.categorical_crossentropy(y_true, y_pred) + K.cast_to_floatx(0.01) * K.abs(year_pred - year_true))


def ultimate_loss_function(y_true, y_pred):
    return (K.categorical_crossentropy(y_true, y_pred) +
            K.cast_to_floatx(1.0)*K.mean(K.abs(y_pred - y_true), axis=-1))

def ultimate_loss_function_100(y_true, y_pred):
    return (K.categorical_crossentropy(y_true, y_pred) +
            K.cast_to_floatx(100.0)*K.mean(K.abs(y_pred - y_true), axis=-1))

def ultimate_loss_function_200(y_true, y_pred):
    return (K.categorical_crossentropy(y_true, y_pred) +
            K.cast_to_floatx(200.0)*K.mean(K.abs(y_pred - y_true), axis=-1))

def ultimate_loss_function_500(y_true, y_pred):
    return (K.categorical_crossentropy(y_true, y_pred) +
            K.cast_to_floatx(500.0)*K.mean(K.abs(y_pred - y_true), axis=-1))

def ultimate_loss_function_1000(y_true, y_pred):
    return (K.categorical_crossentropy(y_true, y_pred) +
            K.cast_to_floatx(1000.0)*K.mean(K.abs(y_pred - y_true), axis=-1))

def ultimate_loss_function_5000(y_true, y_pred):
    return (K.categorical_crossentropy(y_true, y_pred) +
            K.cast_to_floatx(5000.0)*K.mean(K.abs(y_pred - y_true), axis=-1))



def load(image_path):
	#TODO: load image and process if you want to do any
	img=imread(image_path)
	return img
	
class Predictor:
	DATASET_TYPE = 'yearbook'

	# FRZN: load the module here for the whole class to avoid loading it for every prediction
	pdb.set_trace()
	model_name = "m_2017-10-06_02:10_inceptionv3_categorical_crossentropy_adam_lr0.001_epochs50_regnone_decay0.0_ft.model"
	target_size = (299,299) # need to provide this according to the model
	print ("loading model ....")
	model = load_model("../model/fitted_models/" + model_name)
	print ("model loaded")

	# baseline 1 which calculates the median of the train data and return each time
	def yearbook_baseline(self):
		# Load all training data
		train_list = listYearbook(train=True, valid=False)

		# Get all the labels
		years = np.array([float(y[1]) for y in train_list])
		med = np.median(years, axis=0)
		return [med]

	# Compute the median.
	# We do this in the projective space of the map instead of longitude/latitude,
	# as France is almost flat and euclidean distances in the projective space are
	# close enough to spherical distances.
	def streetview_baseline(self):
		# Load all training data
		train_list = listStreetView(train=True, valid=False)

		# Get all the labels
		coord = np.array([(float(y[1]), float(y[2])) for y in train_list])
		xy = coordinateToXY(coord)
		med = np.median(xy, axis=0, keepdims=True)
		med_coord = np.squeeze(XYToCoordinate(med))
		return med_coord

	def predict(self, image_path):

		img = load(image_path)

		#TODO: load model, need to change this to our best model

		#TODO: predict model and return result either in geolocation format or yearbook format
		# depending on the dataset you are using
		


		if self.DATASET_TYPE == 'geolocation':
			result = self.streetview_baseline() #for geolocation
		elif self.DATASET_TYPE == 'yearbook':
			# FRZN: the below line corresponding to baseline has been commented out
			# result = self.yearbook_baseline() #for yearbook
			result = predict.predict_img(self.model, img, self.target_size)
		return result
		
	


