import os
import pandas as pd
import argparse
import operator
import random

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", help="increase output verbosity")
#parser.add_argument('-l', "--label", type=int, help="the num of labels")
parser.add_argument("-p", "--path", type=str, help="path of files")
tempf_path = 'preshuffle'
train_path = 'train'
test_path = 'test'
	
def datashuffle(args):
	file_list = os.listdir(args.path)
	length = len(file_list)
	li=list(range(0, length))
	random.shuffle(li)	

	if not os.path.exists(train_path):
		os.mkdir(train_path)
	if not os.path.exists(test_path):
		os.mkdir(test_path)     
	if not os.path.exists(tempf_path):
		os.mkdir(tempf_path)    

	num_train = int(length*0.8)

	for i, file in enumerate(file_list):
		
		tempfile_path = os.path.join(tempf_path, str(li[i]) + '.txt')
	
		pref = open(os.path.join(args.path, file), 'r')
		tempf = open(tempfile_path, 'w')
		for line in pref:
			tempf.write(line)
		print('temp complete %s', file)
		pref.close() 
		tempf.close( )
		
	file_list = os.listdir(tempf_path)
	for i, file in enumerate(file_list):
		
		if i < num_train:
			afterf_path = os.path.join(train_path, str(li[i]) + '.txt')
		else:
			afterf_path = os.path.join(test_path, str(li[i]) + '.txt')
		
		pref = open(os.path.join(tempf_path, file), 'r')
		afterf = open(afterf_path, 'w')
		for line in pref:
			afterf.write(line)
		print('complete %s', file)
		pref.close() 
		afterf.close( )





if __name__ == '__main__':
	args = parser.parse_args()
	datashuffle(args)





