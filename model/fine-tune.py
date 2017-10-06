import os
import sys
import glob
import argparse
import numpy as np
import matplotlib     #i was getting the following error in plot_training()  https://github.com/matplotlib/matplotlib/issues/3466
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras import __version__
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.vgg16 import VGG16
from keras.applications import vgg16, vgg19, inception_v3, xception, resnet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import SGD
from keras import regularizers
from keras import losses
import keras.losses

import keras.backend as K
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint, EarlyStopping
from predict import predict

import pandas as pd
from shutil import copyfile

#from keras.utils import plot_model
import code  # https://www.digitalocean.com/community/tutorials/how-to-debug-python-with-an-interactive-console
import datetime
import traceback

#default for inceptionv3
ARCHITECTURE = "inceptionv3"
# IM_WIDTH, IM_HEIGHT = 299, 299
NB_EPOCHS = 10
BAT_SIZE = 128   
LEARNING_RATE = 1e-4
# FC_SIZE = 1024
# NB_LAYERS_TO_FREEZE = 172
# LAMBDA = K.cast_to_floatx(1.0)


def mean_L1_distance(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def min_L1_distance(y_true, y_pred):
    return K.min(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def max_L1_distance(y_true, y_pred):
    return K.max(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def std_L1_distance(y_true, y_pred):
    return K.std(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)





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



keras.losses.categorical_crossentropy_mean_squared_error_1 = categorical_crossentropy_mean_squared_error_1
keras.losses.categorical_crossentropy_mean_squared_error_01 = categorical_crossentropy_mean_squared_error_01
keras.losses.categorical_crossentropy_mean_squared_error_001 = categorical_crossentropy_mean_squared_error_001



def test_loss():
    np.random.seed(1)
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))

    print(K.eval(y_a).shape)
    print(K.eval(y_b).shape)

    print(K.eval(y_a))
    # print(K.eval(K.abs(K.argmax(y_a,axis = -1))).shape)
    # print (K.eval(K.abs(K.argmax(y_a,axis = -1))))
    output1 = losses.categorical_crossentropy(y_a, y_b)
    output2 = pure_mean_squared_error(y_a, y_b)
    output = categorical_crossentropy_mean_squared_error(y_a, y_b)
    # output_mse = pure_mean_squared_error(y_a, y_b)
    print('mean_L1:')
    print(K.eval(output).shape)
    print('cross:', K.eval(output1))
    print('mse: ', K.eval(output2))

    print('total: ', K.eval(output))
    # print('total_mse:',K.eval(output_mse))
    # print('cross_entropy:', K.eval(output1))
    # print('mse: ', K.eval(output2))


    assert K.eval(output).shape == (6,)








def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model, base_model, optimizer_in, loss_in, learning_rate, decay  ):
    """Freeze all layers and compile the model"""  # Transfer learning: freeze all but the penultimate layer and re-train the last Dense layer
    print('Number of trainable weight tensors '
      'before freezing the conv base:', len(model.trainable_weights))
    
    for layer in base_model.layers:
        layer.trainable = False
        
    print('Number of trainable weight tensors '
      'after freezing the conv base:', len(model.trainable_weights))
    
    if optimizer_in == 'rmsprop':
        optimizer_tl = optimizers.RMSprop(lr = learning_rate, decay=decay)
    elif optimizer_in == 'adam':
        optimizer_tl = optimizers.Adam(lr = learning_rate, decay = decay)
    elif optimizer_in == 'sgd':
        optimizer_tl = optimizers.SGD(lr = learning_rate, momentum=9.0, nesterov=True)
    elif optimizer_in == 'adagrad':
        optimizer_tl = optimizers.Adagrad(lr = learning_rate)


    model.compile(optimizer = optimizer_tl,
                  loss = loss_in,
                  metrics=['acc', 'top_k_categorical_accuracy', mean_L1_distance, min_L1_distance, max_L1_distance])


def add_new_last_layer(base_model, nb_classes, FC_SIZE, regularizer, reg_rate):
    """Add last layer to the convnet

    Args:
      base_model: keras model excluding top
      nb_classes: # of classes    #passing in 104 from Input, but does this need to be 1000 since image net used that ( or actually number of classes found in train set)?

    Returns:
      new keras model with last layer
    """
    inlayer = base_model.input
    x = base_model.output

    # code.interact(local=locals())
    # print("current output Lastlayer x.shape: ")
    # print(x)				   #Tensor("mixed10/concat:0", shape=(?, ?, ?, 2048), dtype=float32)
    # print(x.shape)			   #(?, ?, ?, 2048)

    x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D converts the MxNxC tensor output into a 1xC tensor where C is the # of channels.
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init  a fully-connected Dense layer of size 1024
    # print("after pooling, dense Lastlayer x.shape: ")
    # print(x)				   #Tensor("dense_1/Relu:0", shape=(?, 1024), dtype=float32)
    # print(x.shape)                           #(?, 1024)

    # new softmax layer on the output to squeeze the values between [0,1]
    if regularizer == "none":
        predictions = Dense(nb_classes, activation='softmax')(x)
    else:
        if regularizer == "L1":
            print("using L1 regularization")
            #see https://keras.io/regularizers/
            predictions = Dense(nb_classes, activation='softmax', kernel_regularizer=regularizers.l1(reg_rate) )(x)
        
    # print("predictions.shape: ")
    print("PREDICTIONS need to be in [0,1].  nb_classes: ", nb_classes, " should be the size of your last layer")
    print(predictions)
    print(predictions.shape)  # (?, 0)

    model = Model(inputs=base_model.input,
                  outputs=predictions)  # UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=Tensor("in..., outputs=Tensor("de...)`
    # fixed via:  https://github.com/fchollet/keras/issues/7602  , change input= to inputs= , output=  to outputs=
    return model


def setup_to_finetune(model, LAYER_FROM_FREEZE, NB_LAYERS_TO_FREEZE, optimizer_in, loss_in, learning_rate, decay):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.  #Fine-tuning: un-freeze the lower convolutional layers and retrain more layers

    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

    Args:
      model: keras model
    """

    print('Number of trainable weight tensors '
      'before starting the fine-tuning step:', len(model.trainable_weights))

    # Feature to unfreeze part of network from LAYER_FROM_FREEZE to the end
    if(LAYER_FROM_FREEZE != ''):
        model.trainable = True

        set_trainable = False

        for layer in model.layers:
            if layer.name == LAYER_FROM_FREEZE:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    else:
        for layer in model.layers[:NB_LAYERS_TO_FREEZE]:
            layer.trainable = False
        for layer in model.layers[NB_LAYERS_TO_FREEZE:]:
            layer.trainable = True
        
    print('Number of trainable weight tensors '
      'during the fine-tuning step:', len(model.trainable_weights))

    if optimizer_in == 'rmsprop':
        optimizer_ft = optimizers.RMSprop(lr = learning_rate/10, decay=decay)
    elif optimizer_in == 'adam':
        optimizer_ft = optimizers.Adam(lr = learning_rate/10, decay=decay)
    elif optimizer_in == 'sgd':
        optimizer_ft = optimizers.SGD(lr = learning_rate/10, momentum=9.0, nesterov=True)
    elif optimizer_in == 'adagrad':
        optimizer_ft = optimizers.Adagrad(lr = learning_rate/10)
      
    # We should use lower learning rate when fine-tuning. learning_rate /10 is a good start.
    model.compile(optimizer=optimizer_ft, loss=loss_in,
                  metrics=['acc', 'top_k_categorical_accuracy', mean_L1_distance, min_L1_distance, max_L1_distance])

def confusion_matrix(model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:,actual_class]==1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class==predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix)==len(truth)
    assert np.sum(confusion_matrix)==np.sum(truth)
    return confusion_matrix

def train(args):

    """Use transfer learning and fine-tuning to train a network on a new dataset"""

    # 0. CREATE EXPECTED FOLDER STRUCTURE
    # take yearbook_train.csv and generate new training folder which makes folders for each year present (treating women/men as same) and copies the files over to them!
    if os.path.isdir(args.data_dir + "/" + args.input_dir + "_" + args.model_name) == False:
        create_folder_with_classes(args.data_dir, args.data_dir + "/" + args.input_dir,
                                   args.input_dir + "_" + args.model_name, args.data_dir + "/" + args.train_file)

    # take yearbook_valid.csv and generate new validation folder which leaves F/  M/  but within each makes folders for each year present and copies the files over to them!
    if os.path.isdir(args.data_dir + "/" + args.valid_dir + "_" + args.model_name) == False:
        create_folder_with_classes(args.data_dir, args.data_dir + "/" + args.valid_dir,
                                   args.valid_dir + "_" + args.model_name, args.data_dir + "/" + args.valid_file)

    nb_train_samples = get_nb_files(args.data_dir + "/" + args.input_dir + "_" + args.model_name)  # 22840
    print("Looking in ", args.data_dir + "/" + args.input_dir + "_" + args.model_name + "/*")
    nb_classes = len(glob.glob(
        args.data_dir + "/" + args.input_dir + "_" + args.model_name + "/*"))  # 104              #1905 - 2013, you would expect 109, but there is no 1907, 1917, 1918, 1920, 1921
    nb_val_samples = get_nb_files(args.data_dir + "/" + args.valid_dir + "_" + args.model_name)  # 5009
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    LAMBDA = int(args.lambda_val)
    decay = float(args.decay)

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

    if args.input_dir == "train_sub":
        results = ['1930','1940','1950','1960','1970','1980','1990','2000']

    print("nb_train_samples: ", nb_train_samples)
    print("nb_classes: ", nb_classes)
    print("nb_val_samples: ", nb_val_samples)
    print("nb_epoch: ", nb_epoch)
    print("batch_size: ", batch_size)
    print("decay: ", batch_size)
    print("LAMBDA: ", batch_size)

    # SET DEFAULTS BASED ON ARTCHITECTURE
    ARCHITECTURE = args.model_name
    if args.model_name == "inceptionv3":
        IM_WIDTH, IM_HEIGHT = 299, 299 
        FC_SIZE = 1024  # should this be 2048 as opposed to 1024.. give it a try
        LAYER_FROM_FREEZE = ''
        NB_LAYERS_TO_FREEZE = 172
        # setup model
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
        # print(base_model.summary())
        preprocess_input = inception_v3.preprocess_input

    if args.model_name == "VGG16":
        IM_WIDTH, IM_HEIGHT = 224, 224
        FC_SIZE = 256
        LAYER_FROM_FREEZE = 'block5_conv1'
        NB_LAYERS_TO_FREEZE = None
        # setup model
        base_model = vgg16.VGG16(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
        # print(base_model.summary())
        preprocess_input = vgg16.preprocess_input


    if args.model_name == "VGG19":
        IM_WIDTH, IM_HEIGHT = 224, 224
        FC_SIZE = 256
        LAYER_FROM_FREEZE = 'block5_conv1'
        NB_LAYERS_TO_FREEZE = None
        # setup model
        base_model = vgg19.VGG19(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
        # print(base_model.summary())
        preprocess_input = vgg19.preprocess_input


    if args.model_name == "Xception":
        IM_WIDTH, IM_HEIGHT = 299, 299
        FC_SIZE = 256
        LAYER_FROM_FREEZE = 'block11_sepconv1_act'
        NB_LAYERS_TO_FREEZE = None
        # setup model
        base_model = xception.Xception(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
        # print(base_model.summary())
        preprocess_input = xception.preprocess_input



    if args.model_name == "ResNet50":
        IM_WIDTH, IM_HEIGHT = 224, 224
        FC_SIZE = 1024
        LAYER_FROM_FREEZE = 'res4a_branch2a'
        NB_LAYERS_TO_FREEZE = None
        # setup model
        base_model = resnet50.ResNet50(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
        # print(base_model.summary())
        preprocess_input = resnet50.preprocess_input





    # 1. PREPROCESS THE IMAGES WE HAVE
    # data prep  #https://keras.io/preprocessing/image/
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # test_datagen = ImageDataGenerator(
    #     preprocessing_function=preprocess_input,
    #     rotation_range=30,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True
    # )

    # Amin: I don't think the validation set shoud be augmented
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    # flow_from_directory(directory): Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.
    # Arguments:
    # - directory: path to the target directory.
    #   It should contain one subdirectory per class. Any PNG, JPG, BMP or PPM images inside each of the subdirectories directory tree will be included in the generator.
    # - target_size: tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized.

    # 2. Go through Training data and Resize/Batch   , same with Valid data
    train_generator = train_datagen.flow_from_directory(
        args.data_dir + "/" + args.input_dir + "_" + args.model_name,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        classes=response_classes
    )

    # label_to_class = train_generator.class_indices
    # class_to_label= {y: x for x, y in label_to_class.items()}
    # print(class_to_label)


    validation_generator = test_datagen.flow_from_directory(
        args.data_dir + "/" + args.valid_dir + "_" + args.model_name,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode= 'categorical',
        classes=response_classes
    )

    model = add_new_last_layer(base_model, nb_classes, FC_SIZE, args.regularizer, args.reg_rate)

    # transfer learning
    setup_to_transfer_learn(model, base_model, args.optimizer, args.loss, float(args.learning_rate),decay)

    datenow = datetime.datetime.today().strftime('%Y-%m-%d_%H:%m')
    output_base = "m_"+datenow+"_"+args.model_name + "_" + args.loss + "_" + args.optimizer + "_lr" + str(args.learning_rate) + "_epochs" + str(nb_epoch) + "_reg"+args.regularizer+"_decay"+str(decay)
    logger_output_tl = output_base+"_tl.log"
    csv_logger_tl = CSVLogger("logs/"+logger_output_tl)
    weights_tl = output_base+"_tl.hdf5"
    checkpointer_tl = ModelCheckpoint(filepath='fitted_models/'+weights_tl, verbose=1, monitor='val_mean_L1_distance', save_best_only=True)
    early_stopping_tl = EarlyStopping(monitor='val_mean_L1_distance', patience=4)
    history_tl = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=nb_train_samples / batch_size,
        validation_data=validation_generator,
        validation_steps=nb_val_samples / batch_size,
        callbacks=[csv_logger_tl,checkpointer_tl, early_stopping_tl],
        class_weight='auto')  # Amin: what is this class_weight?

    output_name = output_base+"_tl.model"
    model.save("fitted_models/" + output_name)

    #code.interact(local=locals())

    print("Save transfer learning plots ...")
    try:
        plot_training(output_name, model, history_tl)
    except Exception as e:
        print(traceback.format_exc())

    # fine-tuning
    setup_to_finetune(model, LAYER_FROM_FREEZE, NB_LAYERS_TO_FREEZE, args.optimizer, args.loss, float(args.learning_rate), decay)

    # Doing transfer learning and then fine-tuning, in that order, will ensure a more stable and consistent training.
    # This is because the large gradient updates triggered by randomly initialized weights could wreck the learned weights in the convolutional base if not frozen.
    # Once the last layer has stabilized (transfer learning), then we move onto retraining more layers (fine-tuning).

    logger_output_ft = output_base+"_ft.log"
    csv_logger_ft = CSVLogger("logs/"+logger_output_ft)
    weights_ft = output_base+"_ft.hdf5"
    checkpointer_ft = ModelCheckpoint(filepath='fitted_models/'+weights_ft, verbose=1, monitor='val_mean_L1_distance', save_best_only=True)
    early_stopping_ft = EarlyStopping(monitor='val_mean_L1_distance', patience=6)
    history_ft = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=nb_train_samples / batch_size,
        validation_data=validation_generator,
        validation_steps=nb_val_samples / batch_size,
        callbacks=[csv_logger_ft,checkpointer_ft,early_stopping_ft],
        class_weight='auto')

    output_name = output_base+"_ft.model"
    print("Save Model "+output_name)
    model.save("fitted_models/"+output_name)

    print("Save fine-tuning plots ...")
    try:
        plot_training(output_name, model, history_ft)
    except Exception as e:
        print(traceback.format_exc())

    acc = history_ft.history['acc']
    val_acc = history_ft.history['val_acc']
    loss = history_ft.history['loss']
    val_loss = history_ft.history['val_loss']


    #Diego:  I'm not sure why we are commenting this out?  Or rather what is a checkpointer?
    #print("Save Model results")
    #results_df = pd.read_csv('model_results.csv')
    #print(len(results_df.index))
    #datenow = datetime.datetime.today().strftime('%Y-%m-%d_%H:%m')
    #res = [ datenow, args.model_name, args.optimizer, args.loss, args.learning_rate, args.nb_epoch, args.batch_size, acc, loss, val_acc, val_loss, output_name ]
    #print(res)
    #results_df.loc[len(results_df.index)+1] = res 
                                             
    #print(results_df)
    #results_df.to_csv("model_results.csv")



def create_folder_with_classes(basef, input_folder, output_folder, trainfile):
    print("Making Folders from " + input_folder + " to " + output_folder)
    train = pd.read_csv(trainfile, names=['imagepath', 'year'], delimiter="\t")
    train['gender'] = [t.split("/")[0] for t in train['imagepath']]
    train['imagepath'] = [t.split("/")[1] for t in train['imagepath']]

    os.system("mkdir " + basef + "/" + output_folder)
    # make women folders and copy over file
    womenyears = sorted(train[train['gender'] == 'F']["year"].unique())

    count_duplicate = 0
    for y in womenyears:
        curfolder = basef + "/" + output_folder + "/" + str(y)
        if os.path.isdir(curfolder) == False:
            os.system("mkdir " + curfolder)
        imgs = train[(train["year"] == y) & (train["gender"] == 'F')]["imagepath"]
        for i in imgs:
            assert (os.path.isfile(input_folder + "/F/" + i))
            if (os.path.isfile(curfolder + "/" + i)):
                count_duplicate += 1
                copyfile(input_folder + "/F/" + i, curfolder + "/d" + i)
            else:
                copyfile(input_folder + "/F/" + i, curfolder + "/" + i)
            assert (os.path.isfile(curfolder + "/" + i))

    # make men folders
    menyears = sorted(train[train['gender'] == 'M']["year"].unique())
    for y in menyears:
        curfolder = basef + "/" + output_folder + "/" + str(y)
        if os.path.isdir(curfolder) == False:
            os.system("mkdir " + curfolder)
        imgs = train[(train["year"] == y) & (train["gender"] == 'M')]["imagepath"]
        for i in imgs:
            assert (os.path.isfile(input_folder + "/M/" + i))
            if (os.path.isfile(curfolder + "/" + i)):
                count_duplicate += 1
                copyfile(input_folder + "/M/" + i, curfolder + "/d" + i)
            else:
                copyfile(input_folder + "/M/" + i, curfolder + "/" + i)
            assert (os.path.isfile(curfolder + "/" + i))

    print("number of duplicate files:", count_duplicate)


def plot_training(modelname,model,history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mean_L1 = history.history['mean_L1_distance']
    val_mean_L1 = history.history['val_mean_L1_distance']

    epochs = range(len(acc))

    
    plt.plot(epochs, acc, 'r.', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.title('Train/val accuracy for '+modelname)
    plt.legend()


    plt.savefig("fitted_models/"+modelname+"_train_val_acc.png")
    plt.close()

    plt.figure()

    plt.plot(epochs, loss, 'r.', label = 'Traning Loss')
    plt.plot(epochs, val_loss, 'r-', label = 'Validation Loss')
    plt.title('Train/val loss for '+modelname)
    plt.legend()

    plt.savefig("fitted_models/"+modelname+"_train_val_loss.png")
    plt.close()


    plt.figure()

    plt.plot(epochs, mean_L1, 'r.', label = 'Traning mean L1 Score')
    plt.plot(epochs, val_mean_L1, 'r-', label = 'Validation mean L1 Score')
    plt.title('Train/val mean L1 Scores for '+modelname)
    plt.legend()

    plt.savefig("fitted_models/"+modelname+"_train_val_mean_L1.png")
    plt.close()

    #this causes conflicts with python 3.6 (it was built on 2.7)
    #plot_model(model, to_file="fitted_models/"+modelname + '_keras.png')

def evaluate(model):

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )



    output = model.predict(train_datagen, batch_size=None, verbose=0, steps=None)


if __name__ == "__main__":
    # SAMPLE CALLs
    # python fine-tune.py --data_dir="../data/yearbook" --model_name="inceptionv3"         #use training set from data/yearbook/train, new images in data/yearbook/train_inception3
    # python fine-tune.py --data_dir="../data/yearbook" --input_dir="train_sub" --valid_dir="valid_sub" --train_file="yearbook_train_small.txt" --valid_file="yearbook_valid_small.txt" --model_name="inceptionv3"
    a = argparse.ArgumentParser()
    a.add_argument("--data_dir", default='../data/yearbook')
    a.add_argument("--input_dir", default="train")
    a.add_argument("--valid_dir", default="valid")
    a.add_argument("--model_name", default="inceptionv3")
    a.add_argument("--train_file", default="yearbook_train.txt")
    a.add_argument("--valid_file", default="yearbook_valid.txt")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--optimizer", default='rmsprop')
    a.add_argument("--loss", default= 'categorical_crossentropy')
    a.add_argument("--learning_rate", default=LEARNING_RATE)
    a.add_argument("--regularizer", default='none')
    a.add_argument("--reg_rate", default=0)
    a.add_argument("--decay", default=0)
    a.add_argument("--lambda_val", default=1)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if (not os.path.exists(args.data_dir)):
        print("directory to data does not exist")
        sys.exit(1)

    model = train(args)
    #evaluate(model, args)  # this is mainly used for confusion matr
    # Using TensorFlow backend.
    # Found 22840 images belonging to 2 classes.
    # Found 5009 images belonging to 2 classes.


    # test_loss()

