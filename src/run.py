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
from keras.applications import vgg16, vgg19, inception_v3, xception, resnet50, imagenet_utils
from PIL import Image
import sys
sys.path.append("../model")
from fine_tune import predict_img


def mean_L1_distance(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def min_L1_distance(y_true, y_pred):
    return K.min(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def max_L1_distance(y_true, y_pred):
    return K.max(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def std_L1_distance(y_true, y_pred):
    return K.std(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

mapping = {
    0: 1905, 1: 1906, 2: 1908, 3: 1909, 4: 1910, 5: 1911, 6: 1912, 7: 1913, 8: 1914, 9: 1915,
    10: 1916, 11: 1919, 12: 1922, 13: 1923, 14: 1924, 15: 1925, 16: 1926, 17: 1927, 18: 1928,
    19: 1929, 20: 1930, 21: 1931, 22: 1932, 23: 1933, 24: 1934, 25: 1935, 26: 1936, 27: 1937,
    28: 1938, 29: 1939, 30: 1940, 31: 1941, 32: 1942, 33: 1943, 34: 1944, 35: 1945, 36: 1946,
    37: 1947, 38: 1948, 39: 1949, 40: 1950, 41: 1951, 42: 1952, 43: 1953, 44: 1954, 45: 1955,
    46: 1956, 47: 1957, 48: 1958, 49: 1959, 50: 1960, 51: 1961, 52: 1962, 53: 1963, 54: 1964,
    55: 1965, 56: 1966, 57: 1967, 58: 1968, 59: 1969, 60: 1970, 61: 1971, 62: 1972, 63: 1973,
    64: 1974, 65: 1975, 66: 1976, 67: 1977, 68: 1978, 69: 1979, 70: 1980, 71: 1981, 72: 1982,
    73: 1983, 74: 1984, 75: 1985, 76: 1986, 77: 1987, 78: 1988, 79: 1989, 80: 1990, 81: 1991,
    82: 1992, 83: 1993, 84: 1994, 85: 1995, 86: 1996, 87: 1997, 88: 1998, 89: 1999, 90: 2000,
    91: 2001, 92: 2002, 93: 2003, 94: 2004, 95: 2005, 96: 2006, 97: 2007, 98: 2008, 99: 2009,
    100: 2010, 101: 2011, 102: 2012, 103: 2013}

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
	img = Image.open(image_path)  # we need to read the image using PIL.Image
	return img
	
class Predictor:
	DATASET_TYPE = 'yearbook'

	# FRZN: load the module here for the whole class to avoid loading it for every prediction
	model_name = "best_model_so_far_VGG16.hdf5"
	#model_name = "m_2017-10-09_15:51_VGG16_ultimate_loss_function_sgd_lr0.0001_epochs20_regnone_decay0.0_ft.model"
	print ("loading model ....")
	model = load_model("../model/fitted_models/" + model_name)
	print ("model loaded")
	#pdb.set_trace()

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

		# need to import model-name here again
		# Read target_size and preprocess_input 
		if "inceptionv3" in self.model_name:
		    IM_WIDTH, IM_HEIGHT = 299, 299 
		    preprocess_input = inception_v3.preprocess_input

		if "VGG16" in self.model_name:
		    IM_WIDTH, IM_HEIGHT = 224, 224
		    preprocess_input = imagenet_utils.preprocess_input

		if "VGG19" in self.model_name:
		    IM_WIDTH, IM_HEIGHT = 224, 224
		    preprocess_input = imagenet_utils.preprocess_input

		if "Xception" in self.model_name:
		    IM_WIDTH, IM_HEIGHT = 299, 299
		    preprocess_input = xception.preprocess_input

		if "ResNet50" in self.model_name:
		    IM_WIDTH, IM_HEIGHT = 224, 224
		    preprocess_input = resnet50.preprocess_input


		if self.DATASET_TYPE == 'geolocation':
			result = self.streetview_baseline() #for geolocation
		elif self.DATASET_TYPE == 'yearbook':
			# FRZN: the below line corresponding to baseline has been commented out
			# result = self.yearbook_baseline() #for yearbook
			y_pred_hot = predict_img(self.model, img, (IM_HEIGHT, IM_WIDTH), preprocess_input)
			y_pred_year = mapping[np.argmax(y_pred_hot)]
			result = [y_pred_year]
		return result