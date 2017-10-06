import pandas as pd
import argparse
import os

if __name__ == "__main__":
    
    a = argparse.ArgumentParser()
    a.add_argument("--cuda_device", default=3)
    a.add_argument("--conf_file", default='conf.csv')
    a.add_argument("--data_dir", default='../data/yearbook')
    a.add_argument("--input_dir", default="train")
    a.add_argument("--valid_dir", default="valid")
    a.add_argument("--train_file", default="yearbook_train.txt")
    a.add_argument("--valid_file", default="yearbook_valid.txt")
    args = a.parse_args()
    
    #open conf.csv and call fine-tune.py with each configuration
    runs = pd.read_csv(args.conf_file)
    for index, row in runs.iterrows():
        print("On: ",row)
        #print(row.architecture)
        syscmd = "CUDA_VISIBLE_DEVICES="+args.cuda_device+" python fine-tune.py --data_dir='"+args.data_dir+ \
                    "' --input_dir='"+args.input_dir+"' --valid_dir='"+args.valid_dir+\
                    "' --train_file='"+args.train_file+"' --valid_file='"+args.valid_file+\
                    "' --model_name='"+row.architecture+"' --optimizer='"+row.optimizer+\
                    "' --loss='"+row.loss+"' --learning_rate="+str(row.learning_rate)+\
                    " --nb_epoch="+str(row.epochs)+" --batch_size="+str(row.batch_size)+\
                    " --regularizer='"+str(row.regularizer)+"' --reg_rate="+str(row.batch_size)+\
                    " --decay='"+str(row.decay)+"' --lambda_val="+str(row.lambda_val)

        print(syscmd)
        os.system(syscmd)
        
'''
mimic old sample call:
CUDA_VISIBLE_DEVICES=3 python fine-tune.py --data_dir="../data/yearbook" --input_dir="train" --valid_dir="valid" --train_file="yearbook_train.txt" --valid_file="yearbook_valid.txt" --model_name="inceptionv3"

mimic new call for multi experiments
CUDA_VISIBLE_DEVICES=3 python run_experiments.py --data_dir="../data/yearbook" --input_dir="train_sub" --valid_dir="valid_sub" --train_file="yearbook_train_sub.txt" --valid_file="yearbook_valid_sub.txt" --model_name="inceptionv3"

using defaults:
CUDA_VISIBLE_DEVICES=3 python run_experiments.py

sample row in conf:
architecture                     inceptionv3
optimizer                            rmsprop
loss                categorical_crossentropy
learning_rate                          1e-05
epochs                                     3
batch_size                               128
layers_to_freeze                         172
Name: 0, dtype: object
'''
