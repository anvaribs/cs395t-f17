from os import path
import util
import numpy as np
import argparse
from skimage.io import imread
from util import *
from model import *
import csv

def load(args):
	#TODO:load image and process if you want to do any
	# Need to be parametrized
	# for not this function only loads and returns training set
	img=imread(image_path)

	batch_size = int(args.batch_size)
    IM_WIDTH, IM_HEIGHT = 299, 299
	# for now need to force classes of validation to be same of train somehow
    response_classes = ['1905', '1906', '1908', '1909', '1910', '1911', '1912', '1913', '1914', '1915', '1916', '1919',
                        '1922', '1923', '1924', '1925', '1926', '1927', '1928', '1929', '1930', '1931', '1932', '1933',
                        '1934', '1935', '1936', '1937', '1938', '1939', '1940', '1941', '1942', '1943', '1944', '1945',
                        '1946', '1947', '1948', '1949', '1950', '1951', '1952', '1953', '1954', '1955', '1956', '1957',
                        '1958', '1959', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',
                        '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981',
                        '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993',
                        '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
                        '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']


    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )


    train_generator = train_datagen.flow_from_directory(
        args.data_dir + "/" + args.input_dir + "_" + args.model_name,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        classes=response_classes
    )


	return train_generator
	
class Predictor:
	DATASET_TYPE = 'yearbook'
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

		#TODO: load model

		#TODO: predict model and return result either in geolocation format or yearbook format
		# depending on the dataset you are using
		if self.DATASET_TYPE == 'geolocation':
			result = self.streetview_baseline() #for geolocation
		elif self.DATASET_TYPE == 'yearbook':
			result = self.yearbook_baseline() #for yearbook
		return result
		
	


