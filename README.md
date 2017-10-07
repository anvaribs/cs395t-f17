# Yearbook/ Geolocation
This is a starter for implementing yearbook or geolocation project.

## Dependencies
 * python (tested with python2.7)
 * numpy
 * sklearn
 * matplotlib + basemap

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
	geo
		train
			000001.JPG
			...
		valid
			...
		test
			...
		geo_train.txt
		geo_valid.txt
		geo_test.txt
model
	TODO: put your final model file in the folder
src
	TODO: modify load and predict function in run.py
	grade.py
	run.py
	util.py
output
	TODO: output the yearbook/geo test file
	geo_test_label.txt
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

2. For Geolocation validation data
```
cd src &&  python grade.py --DATASET_TYPE=geolocation --type=valid
```

### Generating Test Label for project submission
1. For yearbook testing data
```
cd src &&  python grade.py --DATASET_TYPE=yearbook --type=test
```

2. For Geolocation testing data
```
cd src &&  python grade.py --DATASET_TYPE=geolocation --type=test
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



12) port forwarding on microdeep

https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh
- ON MICRODEEP:    ipython notebook --no-browser --port=8890
- ON LAPTOP: ssh -N -p 52617 -f -L localhost:8891:localhost:8890 farzan15@microdeep.ece.utexas.edu
- ON LAPTOP CHROME BROWSER:  localhost:8891
