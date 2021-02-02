import numpy as np
import os

import argparse

def main(params):

	path = params.DATASET + '/Best/'
	all_folder = os.listdir(path)
	for folder in all_folder:
		all_pth = os.listdir(path + folder)
		# if len(all_pth) != 10:
		# 	print folder, 'error', str(len(all_pth))
		# 	continue
		num = 0.0
		for pth in all_pth:
			num += float(pth.split('_')[-1].split('.')[0])

		print(folder, str(num / len(all_pth)))
	
def parse_args():
	parser = argparse.ArgumentParser(description='Low shot benchmark')
	parser.add_argument('--DATASET', default="IndianPines", type=str) #KSC, PaviaU, IndianPines, Botswana,    !!PaviaC

	return parser.parse_args()

if __name__ == '__main__':
	params = parse_args()

	main(params)
