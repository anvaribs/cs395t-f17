import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from skimage.io import imread
import glob
# from tqdm import tqdm

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

import code 

target_size = (299, 299) #fixed size for InceptionV3 architecture 


def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  import pdb; pdb.set_trace()
  return preds[0]

def evaluate_model():
  """Makes predictions on input images and calls the conf_matrix
  ARGS:


  Returns:
  """

  target_size = (299, 299) #fixed size for InceptionV3 architecture 
  # modelname = 
  model = load_model("./fitted_models" + modelname)
  output = []

  mylist = dir(hdf5_getters)
  # this is the address on microdeep
  glob_path = '/home/farzan15/cs395t-f17/data/yearbook/A/A/*'
  filepaths = glob.glob(glob_path)
  for filepath in filepaths:
    img = imread(filepath)
    import pdb; pdb.set_trace()
    output.append(predict(model, img, target_size))
  return output


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
  image: PIL image
  preds: list of predicted labels and their probabilities
  """
  print(preds)
  code.interact(local=locals())
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  #labels = ("cat", "dog")
  labels = [i + 2000  for i in range(14) ]
  #plt.barh([0, 1], preds, alpha=0.5)
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  #plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()

  savefig("pred.png")

if __name__=="__main__":
  #python predict.py --image dog.001.jpg --model dc.model
  #python predict.py --image_url https://goo.gl/Xws7Tp --model dc.model

  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_url", help="url to image")
  a.add_argument("--model")
  args = a.parse_args()

  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  model = load_model(args.model)
  print(model.summary())
  if args.image is not None:
    img = Image.open(args.image)
    preds = predict(model, img, target_size)
    print(preds)
    #plot_preds(img, preds)

  if args.image_url is not None:
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    #plot_preds(img, preds)

