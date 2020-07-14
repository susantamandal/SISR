import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import sys
import cv2 as cv
import numpy as np
import tensorlayer as tl
import tensorflow as tf2
import tensorflow.compat.v1 as tf1
 
def get_train_dataset(CONFIG):

	number_files = len(os.listdir(CONFIG.dir_train_in)) 
	CONFIG.no_of_batches = number_files//CONFIG.batch_size
	file_list = tf1.data.Dataset.list_files(CONFIG.dir_train_in + '/*')
  
	def mappings(img_path):
		img = tf1.io.read_file(img_path)
		img = tf1.image.decode_png(img)
		lrimg = tf2.image.resize_with_pad(img, 128, 128, antialias=True)
		hrimg = tf2.image.resize_with_pad (img, 256, 256, antialias=True)
		return lrimg, hrimg	
	dataset = file_list.map(mappings)
	dataset = dataset.shuffle(50)	#shuffle_buffer_size=128
	dataset = dataset.prefetch(buffer_size=2)
	dataset = dataset.batch(CONFIG.batch_size) 
	return dataset

'''def get_validation_data(CONFIG):
	file_list = tf1.data.Dataset.list_files(CONFIG.dir_val_in+'/*')
    #hr_file_list = tf1.data.Dataset.list_files(CONFIG.dir_val_target+'/*')
	def mappings(img_path):
		img = tf1.io.read_file(img_path)
		img = tf1.image.decode_png(img)
		lrimg = tf2.image.resize_with_pad(img, 128, 128, antialias=True)
		#hrimg = tf2.image.resize_with_pad (img, 256, 256, antialias=True)
		return lrimg
	dataset = file_list.map(mappings)
	dataset = dataset.shuffle(50)	#shuffle_buffer_size=128
	dataset = dataset.prefetch(buffer_size=2)
	dataset = dataset.batch(CONFIG.batch_size)
	#for step, lrimg in enumerate(dataset):
	#	print(step, lrimg.shape)
	return dataset
'''    
def generate_bicubic_samples(lrimg_list):
	#print(type(lrimg_list),lrimg_list.shape)
	#lrimg_list=lrimg_list.numpy()
	#print(type(lrimg_list),lrimg_list.shape)
	desired_size = (2*lrimg_list.shape[1],2*lrimg_list.shape[2])
	bcimg_list = np.array([cv.resize(lrimg_list[i], desired_size,interpolation = cv.INTER_CUBIC) for i in range(lrimg_list.shape[0]) ] )
	#print(type(bcimg_list),bcimg_list.shape)	
	return bcimg_list[:,:,:,np.newaxis] 
  
def PSNR(hrimg_list, bcimg_list, opimg_list):

	hrimg_list = cast_uint8(hrimg_list)
	bcimg_list = cast_uint8(bcimg_list)
	opimg_list = cast_uint8(opimg_list)
	print(opimg_list.shape, bcimg_list.shape, hrimg_list.shape)
	bicubic_psnr = tf1.image.psnr( hrimg_list, bcimg_list, max_val=255)
	model_psnr = tf1.image.psnr( hrimg_list, opimg_list, max_val=255)
	return bicubic_psnr, model_psnr

def SSIM(hrimg_list, bcimg_list, opimg_list):

	hrimg_list = cast_uint8(hrimg_list)
	bcimg_list = cast_uint8(bcimg_list)
	opimg_list = cast_uint8(opimg_list)
	hrimg_list = tf1.convert_to_tensor(hrimg_list)
	bcimg_list = tf1.convert_to_tensor(bcimg_list)
	opimg_list = tf1.convert_to_tensor(opimg_list)
	print(opimg_list.shape, bcimg_list.shape, hrimg_list.shape)
	bicubic_ssim = tf1.image.ssim( hrimg_list, bcimg_list, max_val=255)
	model_ssim = tf1.image.ssim( hrimg_list, opimg_list, max_val=255)
	return bicubic_ssim, model_ssim

def cast_uint8(img):
	img = img/np.amax(img)
	img = (img * 255).astype(np.uint8)
	return img


def refresh_folder(dir_path):

	if tl.files.folder_exists(dir_path):
	  tl.files.del_folder(dir_path)
	tl.files.exists_or_mkdir(dir_path)
	print(dir_path+' refreshed!')

def remove_file(file_path):

	if not tl.files.file_exists(file_path):
		print('\033[91m'+ 'No such file exists!' +'\033[0m')
		return
	tl.files.del_file(file_path)
	print(file_path+' removed!')

def copy_images(string):
	
	argv=string.split(" ")
	if not (tl.files.folder_exists(argv[0]) and tl.files.folder_exists(argv[1])) :
		print('\033[91m'+ 'No such folders exist!' +'\033[0m')
		return
	img_file_list = tl.files.load_file_list(path=argv[0], regx='.*.png', printable=False)#.sort(key=tl.files.natural_keys)
	img_file_list.sort(key=tl.files.natural_keys)
	img_file_list=img_file_list[int(argv[2])-1:int(argv[3])]
	img_list = tl.vis.read_images(img_file_list, path=argv[0], n_threads=32)
	for i in range(len(img_file_list)):
		img_path = argv[1]+'/'+img_file_list[i]
		tl.vis.save_image(np.array(img_list[i]), img_path)
		#print('Copy:'+argv[0]+'/'+img_file_list[i]+' To:'+argv[1])
	print('Copy successful!')

def makedataset(string):
	argv=string.split(" ")
	if not (tl.files.folder_exists(argv[0]) and tl.files.folder_exists(argv[1])) :
		print('\033[91m'+ 'No such folders exist!' +'\033[0m')
		return
	img_file_list = tl.files.load_file_list(path=argv[0], regx='.*.png', printable=False)#.sort(key=tl.files.natural_keys)
	img_file_list.sort(key=tl.files.natural_keys)
	img_file_list=img_file_list[int(argv[2])-1:int(argv[3])]
	img_list = tl.vis.read_images(img_file_list, path=argv[0], n_threads=32)
	desired_size = ( lrimg_list.shape[1]//2, img_list.shape[2]//2)
	img_list = np.array([cv.resize(lrimg_list[i],desired_size) for i in range(img_list.shape[0]) ] )
	for i in range(len(img_file_list)):
		img_path = argv[1]+'/'+img_file_list[i]
		tl.vis.save_image(img_list[i], img_path)
	#print('Copy:'+argv[0]+'/'+img_file_list[i]+' To:'+argv[1])
	print('Conversion successful!')

if __name__ == '__main__':
	eval(sys.argv[1]+'("'+sys.argv[2]+'")')
