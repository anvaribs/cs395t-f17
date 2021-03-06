# Yearbook
cs395t project1: implementing yearbook project.  
  * More information at: http://www.philkr.net/cs395t/project1/
  * Report in report.pdf
  * Short presentation in cs395t-F17-proj1-Amin\_Diego\_Farzan.pdf

## Dependencies
 * python (tested with python3.5)
 * numpy
 * sklearn
 * matplotlib + basemap
 * keras

# Project Folder Structure
```
data
	yearbook
		train
			F
				000001.png
				...
			M
				000001.png
				...
		valid
			...
		test
			...
		yearbook_train.txt
		yearbook_valid.txt
		yearbook_test.txt
model
	model source code and final model
src
	grade.py
	run.py
	util.py
output
	yearbook_test_label.txt
```

## Evaluation
### Data setup
Download the data from the link and store it in data folder as described in the folder structure.

### Models
Train the model and put the model in the `Model` folder

### Running the evaluation
It will give the result based on the baseline 1 which is the median of the training image.
1. For yearbook validation data
```
cd src &&  python grade.py --DATASET_TYPE=yearbook --type=valid
```

### Generating Test Label for project submission
1. For yearbook testing data
```
cd src &&  python grade.py --DATASET_TYPE=yearbook --type=test
```

## Submission
1. Put model and generated test_label files in their respective folder.
2. Remove complete data from the data folder.
3. Add readme.md file in your submission.
4. Project should be run from the grade.py file as shown in the evaluation step and should be able to generate the test_label file.





## cs395proj1_install_notes.txt

STEP A: installing anaconda on ubuntu

https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

1) curl -O https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
2) bash Anaconda3-4.2.0-Linux-x86_64.sh
3) source ~/.bashrc


STEP B: create conda environment and get all the dependencies

4) conda create --name tf python=3
5) source activate tf
6) conda install keras
7) conda install matplotlib
8) conda install pandas
9) conda install -c anaconda tensorflow-gpu

STEP C: Train our models using GPUs with "screen" so that you may leave the process running in the background
##https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session

10) screen -S diego395

If USING microdeep server run either 11a for one experiment or 11b for multiple ( both from within model/ folder )
11a) CUDA_VISIBLE_DEVICES=3 python fine-tune.py --data_dir="../data/yearbook" --input_dir="train"
--valid_dir="valid" --train_file="yearbook_train.txt" --valid_file="yearbook_valid.txt"
--model_name="inceptionv3" 

11a) CUDA_VISIBLE_DEVICES=3 python run_experiments.py

ctrl-A ctrl-D  #to detach process and leave it running in background
screen -r diego395   #to get back to screen session to check on process

nvidia-smi  # to see gpu usage on microdeep
top         # to see cpu usage on microdeep



## How ro predict based on a model and a dataset (train or valid)
1) run the following from the model directory:

python fine-tune.py --make_prediction="yes" --pred_dataset="train" --pred_model="m_2017-10-06_02:10_inceptionv3_categorical_crossentropy_adam_lr0.001_epochs50_regnone_decay0.0_ft.model"


NOTE:
- what this does is that it stores y_pred_train.csv and y_true_train.csv in the plot directory. if you run it for test it will produce the corresponding predictions
- the model should be in fitted_models
- depending on the dataset you are interested in, you should choose train or valid
- the make_conf_mat should be yes if you want to make predictions
- note that depending on which model you are using the prediction for, you must provide the size of the input image as pred_target_size exactly in the format of the example above
- if you want to store the predictions for multiple models put the current predictions in another folder and rename them cause they will be overwriteen


2) The next step is to use the model/plot/conf_matrix.ipynb for post processing

once you are in the ipython, you need to specify the dataset of interest, if the predictions for that dataset are stored, it will read it and find confusion matrix and plot it. It will also produce min, mean, max L1 
