import keras.metrics
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2



def mean_L1_distance(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def min_L1_distance(y_true, y_pred):
    return K.min(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def max_L1_distance(y_true, y_pred):
    return K.max(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def std_L1_distance(y_true, y_pred):
    return K.std(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)


keras.metrics.min_L1_distance= min_L1_distance
keras.metrics.max_L1_distance= max_L1_distance
keras.metrics.mean_L1_distance= mean_L1_distance





mapping = {0: '1905', 1: '1906', 2: '1908', 3: '1909', 4: '1910', 5: '1911', 6: '1912', 7: '1913', 8: '1914', 9: '1915',
           10: '1916', 11: '1919', 12: '1922', 13: '1923', 14: '1924', 15: '1925', 16: '1926', 17: '1927', 18: '1928',
           19: '1929', 20: '1930', 21: '1931', 22: '1932', 23: '1933', 24: '1934', 25: '1935', 26: '1936', 27: '1937',
           28: '1938', 29: '1939', 30: '1940', 31: '1941', 32: '1942', 33: '1943', 34: '1944', 35: '1945', 36: '1946',
           37: '1947', 38: '1948', 39: '1949', 40: '1950', 41: '1951', 42: '1952', 43: '1953', 44: '1954', 45: '1955',
           46: '1956', 47: '1957', 48: '1958', 49: '1959', 50: '1960', 51: '1961', 52: '1962', 53: '1963', 54: '1964',
           55: '1965', 56: '1966', 57: '1967', 58: '1968', 59: '1969', 60: '1970', 61: '1971', 62: '1972', 63: '1973',
           64: '1974', 65: '1975', 66: '1976', 67: '1977', 68: '1978', 69: '1979', 70: '1980', 71: '1981', 72: '1982',
           73: '1983', 74: '1984', 75: '1985', 76: '1986', 77: '1987', 78: '1988', 79: '1989', 80: '1990', 81: '1991',
           82: '1992', 83: '1993', 84: '1994', 85: '1995', 86: '1996', 87: '1997', 88: '1998', 89: '1999', 90: '2000',
           91: '2001', 92: '2002', 93: '2003', 94: '2004', 95: '2005', 96: '2006', 97: '2007', 98: '2008', 99: '2009',
           100: '2010', 101: '2011', 102: '2012', 103: '2013'}








def class_activation_heatmap(model_path, img_path):
    model = load_model(model_path)
    print(model.summary())

    # `img` is a PIL image of size 224x224
    img = image.load_img(img_path, target_size=(224, 224))

    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = preprocess_input(x)

    print(x.shape)

    plt.imshow(x[0])
    plt.show()

    preds = model.predict(x)
    print(preds.shape)

    print('year prediction: ', mapping[np.argmax(preds[0])])

    # This is the "predicted" entry in the prediction vector
    predicted_output = model.output[:, np.argmax(preds[0])]

    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('block5_conv3')

    # This is the gradient of the "predicted output" class with regard to
    # the output feature map of `block5_conv3`
    grads = K.gradients(predicted_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    #let's plot the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()



    # superimpose the original image with the heatmap
    # We use cv2 to load the original image
    img = cv2.imread(img_path)

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img

    # Save the image to disk
    cv2.imwrite(img_path + '/superimposed.png', superimposed_img)



if __name__ == '__main__':
    img_path = '../data/yearbook/train_inceptionv3/1910/000644.png'
    model_path = './fitted_models/VGG16_categorical_crossentropy_rmsprop_lr0.0001_epochs10_regnone_ft.model'

    class_activation_heatmap(model_path, img_path)
