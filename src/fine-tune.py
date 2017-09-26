import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

import pandas as pd
from shutil import copyfile

import code #https://www.digitalocean.com/community/tutorials/how-to-debug-python-with-an-interactive-console

IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024         #should this be 2048 as opposed to 1024.. give it a try
NB_IV3_LAYERS_TO_FREEZE = 172


def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""   #Transfer learning: freeze all but the penultimate layer and re-train the last Dense layer
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])    #just categorical_crossentropy,  maybe could add sparse_ to it
  # A target array with shape (32, 70) was passed for an output of shape (None, 0) while using as loss `categorical_crossentropy`. This loss expects targets to have the same shape as the output.
  #   Your model has an output of shape (10,), however, your outputs have dimension (1,). 
  #   You probably want to convert your y_train to categorical one-hot vectors, ie, via keras.utils.np_utils.to_categorical.


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet

  Args:
    base_model: keras model excluding top
    nb_classes: # of classes    #passing in 104 from Input, but does this need to be 1000 since image net used that ( or actually number of classes found in train set)?

  Returns:
    new keras model with last layer
  """
  inlayer = base_model.input
  x = base_model.output


  #code.interact(local=locals())
  #print("current output Lastlayer x.shape: ")     
  #print(x)				   #Tensor("mixed10/concat:0", shape=(?, ?, ?, 2048), dtype=float32)
  #print(x.shape)			   #(?, ?, ?, 2048)	

  x = GlobalAveragePooling2D()(x)          #GlobalAveragePooling2D converts the MxNxC tensor output into a 1xC tensor where C is the # of channels.
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init  a fully-connected Dense layer of size 1024
  #print("after pooling, dense Lastlayer x.shape: ")
  #print(x)				   #Tensor("dense_1/Relu:0", shape=(?, 1024), dtype=float32)
  #print(x.shape)                           #(?, 1024)

  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer on the output to squeeze the values between [0,1]
  #print("predictions.shape: ")             
  print("PREDICTIONS should need to be in [0,1].  nb_classes: ",nb_classes," should be the size of your last layer")
  print(predictions)                       
  print(predictions.shape)                 #(?, 0)

  model = Model(inputs=base_model.input, outputs=predictions)    #UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=Tensor("in..., outputs=Tensor("de...)`
  #fixed via:  https://github.com/fchollet/keras/issues/7602  , change input= to inputs= , output=  to outputs=
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.  #Fine-tuning: un-freeze the lower convolutional layers and retrain more layers

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def create_folder_with_classes(basef,input_folder,output_folder,trainfile):
  print("Making Folders from "+input_folder+" to "+output_folder)
  train = pd.read_csv(trainfile,names=['imagepath','year'],delimiter="\t")
  train['gender'] = [ t.split("/")[0] for t in train['imagepath']]
  train['imagepath'] = [ t.split("/")[1] for t in train['imagepath']]

  os.system("mkdir "+basef+"/"+output_folder)
  #make women folders and copy over file
  womenyears = sorted(train[train['gender']=='F']["year"].unique())
  for y in womenyears:
    curfolder = basef+"/"+output_folder+"/"+str(y)
    if os.path.isdir(curfolder) == False:
      os.system("mkdir "+curfolder)
    imgs = train[(train["year"] == y) & (train["gender"] == 'F')]["imagepath"]
    for i in imgs:
      copyfile(input_folder+"/F/"+i, curfolder+"/"+i)

  #make men folders
  menyears = sorted(train[train['gender']=='M']["year"].unique())
  for y in menyears:
    curfolder = basef+"/"+output_folder+"/"+str(y)
    if os.path.isdir(curfolder) == False:
      os.system("mkdir "+curfolder)
    imgs = train[(train["year"] == y) & (train["gender"] == 'M')]["imagepath"]
    for i in imgs:
      copyfile(input_folder+"/M/"+i, curfolder+"/"+i)

def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""

  #0. CREATE EXPECTED FOLDER STRUCTURE
  #take yearbook_train.csv and generate new training folder which makes folders for each year present (treating women/men as same) and copies the files over to them!
  if os.path.isdir(args.data_dir+"/"+args.input_dir+"_"+args.model_name) == False:
    create_folder_with_classes(args.data_dir,args.data_dir+"/"+args.input_dir,args.input_dir+"_"+args.model_name,args.data_dir+"/"+args.train_file)

  
  #take yearbook_valid.csv and generate new validation folder which leaves F/  M/  but within each makes folders for each year present and copies the files over to them!
  if os.path.isdir(args.data_dir+"/"+args.valid_dir+"_"+args.model_name) == False:
    create_folder_with_classes(args.data_dir,args.data_dir+"/"+args.valid_dir,args.valid_dir+"_"+args.model_name,args.data_dir+"/"+args.valid_file)

  nb_train_samples = get_nb_files(args.data_dir+"/"+args.input_dir+"_"+args.model_name) #22840
  print("Looking in ", args.data_dir + "/" + args.input_dir+"_"+args.model_name+"/*")
  nb_classes = len(glob.glob(args.data_dir + "/"+args.input_dir+"_"+args.model_name+"/*"))  #104              #1905 - 2013, you would expect 109, but there is no 1907, 1917, 1918, 1920, 1921 
  nb_val_samples = get_nb_files(args.data_dir+"/"+args.valid_dir+"_"+args.model_name) #5009
  nb_epoch = int(args.nb_epoch)
  batch_size = int(args.batch_size)

  print("nb_train_samples: ",nb_train_samples)
  print("nb_classes: ",nb_classes)
  print("nb_val_samples: ",nb_val_samples)
  print("nb_epoch: ",nb_epoch)
  print("batch_size: ",batch_size)

  #1. PREPROCESS THE IMAGES WE HAVE
  # data prep  #https://keras.io/preprocessing/image/
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )
  test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  #flow_from_directory(directory): Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.
  #Arguments:
  #- directory: path to the target directory. 
  #   It should contain one subdirectory per class. Any PNG, JPG, BMP or PPM images inside each of the subdirectories directory tree will be included in the generator. 
  #- target_size: tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized.

  #2. Go through Training data and Resize/Batch   , same with Valid data 
  train_generator = train_datagen.flow_from_directory(    
    args.data_dir + "/" +args.input_dir+"_"+args.model_name,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  validation_generator = test_datagen.flow_from_directory(
    args.data_dir + "/" + args.valid_dir+"_"+args.model_name,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  # setup model
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

  # transfer learning
  setup_to_transfer_learn(model, base_model)

  '''
  history_tl = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')

  fine-tune.py:193: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  class_weight='auto')

fine-tune.py:193: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras.pre..., validation_data=<keras.pre..., class_weight="auto", steps_per_epoch=6, epochs=3, validation_steps=20)`
  class_weight='auto')

  '''
  history_tl = model.fit_generator(
    train_generator,
    epochs=nb_epoch,
    steps_per_epoch=nb_train_samples/batch_size,
    validation_data=validation_generator,
    validation_steps=nb_val_samples/batch_size,
    class_weight='auto')

 ##  File "//anaconda/envs/tf/lib/python3.6/site-packages/keras/engine/training.py", line 144, in _standardize_input_data
 ##   str(array.shape))
 ##  ValueError: Error when checking target: expected dense_2 to have shape (None, 1) but got array with shape (32, 70)
  
  '''
  File "fine-tune.py", line 209, in train
    class_weight='auto')
  File "//anaconda/envs/tf/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 87, in wrapper
    return func(*args, **kwargs)
  File "//anaconda/envs/tf/lib/python3.6/site-packages/keras/engine/training.py", line 2042, in fit_generator
    class_weight=class_weight)
  File "//anaconda/envs/tf/lib/python3.6/site-packages/keras/engine/training.py", line 1756, in train_on_batch
    check_batch_axis=True)
  File "//anaconda/envs/tf/lib/python3.6/site-packages/keras/engine/training.py", line 1393, in _standardize_user_data
    self._feed_output_shapes)
  File "//anaconda/envs/tf/lib/python3.6/site-packages/keras/engine/training.py", line 297, in _check_loss_and_target_compatibility
    ' while using as loss `' + loss.__name__ + '`. '
  ValueError: A target array with shape (32, 70) was passed for an output of shape (None, 0) while using as loss `categorical_crossentropy`. This loss expects targets to have the same shape as the output.

  ''' 

  # fine-tuning
  setup_to_finetune(model)

  #Doing transfer learning and then fine-tuning, in that order, will ensure a more stable and consistent training. 
  #This is because the large gradient updates triggered by randomly initialized weights could wreck the learned weights in the convolutional base if not frozen. 
  #Once the last layer has stabilized (transfer learning), then we move onto retraining more layers (fine-tuning).

  '''
  history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')
  '''

  history_ft = model.fit_generator(
    train_generator,
    epochs=nb_epoch,
    steps_per_epoch=nb_train_samples/batch_size,
    validation_data=validation_generator,
    validation_steps=nb_val_samples/batch_size,
    class_weight='auto')

  model.save(args.output_model_file)

  if args.plot:
    plot_training(history_ft)


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()


if __name__=="__main__":
  #SAMPLE CALLs 
  #python fine-tune.py --data_dir="../data/yearbook" --model_name="inceptionv3"         #use training set from data/yearbook/train, new images in data/yearbook/train_inception3
  #python fine-tune.py --data_dir="../data/yearbook" --input_dir="train_sub" --valid_dir="valid_sub" --train_file="yearbook_train_small.txt" --valid_file="yearbook_valid_small.txt" --model_name="inceptionv3"
  a = argparse.ArgumentParser()
  a.add_argument("--data_dir")
  a.add_argument("--input_dir",default="train")
  a.add_argument("--valid_dir",default="valid")
  a.add_argument("--model_name",default="inceptionv3")
  a.add_argument("--train_file",default="yearbook_train.txt")
  a.add_argument("--valid_file",default="yearbook_valid.txt")
  a.add_argument("--nb_epoch", default=NB_EPOCHS)
  a.add_argument("--batch_size", default=BAT_SIZE)
  a.add_argument("--output_model_file", default="inceptionv3-ft.model")
  a.add_argument("--plot", action="store_true")

  args = a.parse_args()
  if (not os.path.exists(args.data_dir)): 
    print("directory to data does not exist")
    sys.exit(1)

  train(args)
  #Using TensorFlow backend.
  #Found 22840 images belonging to 2 classes.
  #Found 5009 images belonging to 2 classes.

