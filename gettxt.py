import os
import pandas as pd
import argparse
import operator
import math

#------- python gettxt.py -p C:\\Users\\zhtang\\Desktop\\water\\rawdatafinalnoise

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", help="increase output verbosity")
parser.add_argument('-l', "--label", type=int, help="the num of labels")
parser.add_argument("-f", "--files", nargs='+', type=str, help="list of files")
parser.add_argument("-p", "--path", type=str, help="path of files")

def makedatesets(file_name, start_num):


	# read the file
	file_dir='C:\\Users\\zhtang\\Desktop\\water\\rawdatafinalnoise'
	new_f_dir = 'C:\\Users\\zhtang\\Desktop\\water\\rawdatafinalnoise\\orderd_data'
	file_path = os.path.join(file_dir, file_name)
	# data = pd.read_csv(file_path, low_memory=False)
	# print data.ix[:10]['Day_of_Week']
	item = list()

	# f = open('E:/学习相关/Python/数据样例/用户侧数据/test数据.csv')
	# reader = pd.read_csv(f, sep=',', iterator=True)
	# loop = True
	# chunkSize = 100000
	# chunks = []
	# while loop:
	#     try:
	#         chunk = reader.get_chunk(chunkSize)
	#         chunks.append(chunk)
	#     except StopIteration:
	#         loop = False
	#         print("Iteration is stopped.")
	# df = pd.concat(chunks, ignore_index=True)
	# print(df)

	dframe = pd.read_excel(file_path)
	for index, line in dframe.iterrows():
		print(" new line ============ ** {} **=\n".format(index))
		# print(line)
		# print(line[0], line[1], line[2], line[3])
		line = [line[i] for i in range(len(line))]
		# print([str(line[i]).strip() for i in range(2)])
		if operator.eq([str(line[i]).strip() for i in range(4)], ['-1.0', '-1.0','-1.0','-1.0']):
		# if '-' in str(line[0]):
			print([str(line[i]).strip() for i in range(2)])
			if len(item) > 0:
				txt_name = str(start_num) + '.txt'
				start_num += 1
				new_f_path = os.path.join(new_f_dir, txt_name)
				with open(new_f_path, 'a') as new_f:
					for item_line in item:
						writeline = ','.join(item_line)
						new_f.writelines([writeline, '\n'])
				new_f.close()
			item = list()
		else:
			item.append([str(strnum) for strnum in line])

	# with open(file_path) as f:
	# 	for line in f:
	# 		# print line
			
	# 		if operator.eq(line ,[-1, -1, -1, -1]):
	# 			if len(item) > 0:
	# 				start_num+=1
	# 				txt_name = str(start_num) + '.txt'
	# 				new_f_path = os.path.join(new_f_dir, txt_name)
	# 				with open(new_f_path, 'a') as new_f:
	# 					for item_line in item: 
							# writeline = ','.join(item_line)
							# new_f.writelines([writeline, '\n'])
	# 				new_f.close()
	# 			item = list()
	# 		else :
	# 			item.append(line)
	# f.close()
	return start_num

def isEqual(a, b, relError):
	return (math.fabs(a - b) < relError)

if __name__ == '__main__':
	args = parser.parse_args()
	start_num = 0
	file_list = os.listdir(args.path)
	for file_name in file_list:
		start_num = makedatesets(file_name, start_num)
