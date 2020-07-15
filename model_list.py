import sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorlayer as tl
import tensorflow as tf2
import tensorflow.compat.v1 as tf1
from tensorlayer.layers import (Input, Conv2d, UpSampling2d, Concat, Elementwise, SubpixelConv2d, BatchNorm2d, Flatten, Dense)
from tensorlayer.models import Model

def get_model(model_type, model_id):
	return eval('model_'+model_type+str(model_id)+'()')
	
def model_G1():	##Phase1 Generator

	gamma_init = tf1.random_normal_initializer(1.,0.02)
	w_init = tf1.random_normal_initializer(stddev=0.02)
	fn =  tf1.nn.relu

	##	Input layers
	lr_image = Input((None, 128, 128, 1))	##	(batch_size, height, width, channel)
	hr_image = Input((None, 256, 256, 1))
	
	## 	Feature extracting layers from LR image
	lr_feature_layer_1 = Conv2d(64, (3, 3), (1, 1), act = fn, padding='SAME', W_init=w_init )(lr_image)# Shape(1,256,256,64)
	lr_feature_layer_1 = BatchNorm2d(gamma_init=gamma_init)(lr_feature_layer_1)
	
	lr_feature_layer_2 = SubpixelConv2d(scale=2, n_out_channels=64, act= fn)(lr_feature_layer_1) # Shape(1,256,256,16)

	lr_feature_layer_3 = Conv2d(64, (3, 3), (1, 1), act = fn, padding='SAME', W_init=w_init )(lr_feature_layer_2) # Shape(1,256,256,64)
	lr_feature_layer_3 = BatchNorm2d(gamma_init=gamma_init)(lr_feature_layer_3)
 
	##	Feature extracting layers from HR image

	hr_feature_layer_1 = Conv2d(64, (3, 3), (1, 1), act= fn, padding='SAME', W_init=w_init )(hr_image)# Shape(1,256,256,64)
	hr_feature_layer_1 = BatchNorm2d(gamma_init=gamma_init)(hr_feature_layer_1)
 
	##	Features Merging layers

	merge_layer = Concat(concat_dim = -1  )([lr_feature_layer_3, hr_feature_layer_1]) # Shape(1,256,256,128)

	non_linearity_layer_1 = Conv2d(256, (3, 3), (1, 1), act= fn, padding='SAME', W_init=w_init )(merge_layer)# Shape(1,256,256,256)
	non_linearity_layer_1 = BatchNorm2d(gamma_init=gamma_init)(non_linearity_layer_1)

	non_linearity_layer_2 = Conv2d(128, (3, 3), (1, 1), act= fn, padding='SAME', W_init=w_init )(non_linearity_layer_1) # Shape(1,256,256,128)
	non_linearity_layer_2 = BatchNorm2d(gamma_init=gamma_init)(non_linearity_layer_2)
	
	non_linearity_layer_3 = Elementwise(combine_fn=tf1.add )([non_linearity_layer_2, merge_layer]) # Shape(1,256,256,128)

	## 	Reconstruction layers
	Recon_layer_1 = Conv2d(1, (5, 5), (1, 1), act=fn, padding='SAME', W_init=w_init )(non_linearity_layer_3) # Shape(1,256,256,1)
	Recon_layer_1 = BatchNorm2d(gamma_init=gamma_init)(Recon_layer_1)
	
	Recon_layer_2 = Elementwise(combine_fn=tf1.add )([Recon_layer_1, hr_image]) # Shape(1,256,256,1)

	return Model(inputs=[lr_image, hr_image], outputs=Recon_layer_2)

def model_G2():	##Phase2 Generator

	gamma_init = tf1.random_normal_initializer(1.,0.02)
	w_init = tf1.random_normal_initializer(stddev=0.02)
	fn =  tf1.nn.relu

	##	Input layers
	lr_image = Input((None, 128, 128, 3))	##	(batch_size, height, width, channel)
	hr_image = Input((None, 512, 512, 3))
	
	## 	Feature extracting layers from LR image
	lr_feature_layer_1 = Conv2d(64, (3, 3), (1, 1), act = fn, padding='SAME', W_init=w_init )(lr_image)# Shape(1,256,256,64)
	lr_feature_layer_1 = BatchNorm2d(gamma_init=gamma_init)(lr_feature_layer_1)
	
	lr_feature_layer_2 = SubpixelConv2d(scale=4 , act= fn)(lr_feature_layer_1) # Shape(1,256,256,16)


	##	Feature extracting layers from HR image

	hr_feature_layer_1 = Conv2d(64, (3, 3), (1, 1), act= fn, padding='SAME', W_init=w_init )(hr_image)# Shape(1,256,256,64)
	hr_feature_layer_1 = BatchNorm2d(gamma_init=gamma_init)(hr_feature_layer_1)
 
	##	Features Merging layers

	merge_layer = Concat(concat_dim = -1  )([lr_feature_layer_2, hr_feature_layer_1]) # Shape(1,256,256,128)

	non_linearity_layer_1 = Conv2d(64, (5, 5), (1, 1), act= fn, padding='SAME', W_init=w_init )(merge_layer)# Shape(1,256,256,256)
	non_linearity_layer_1 = BatchNorm2d(gamma_init=gamma_init)(non_linearity_layer_1)

	 

	## 	Reconstruction layers
	Recon_layer_1 = Conv2d(3, (5, 5), (1, 1), act=fn, padding='SAME', W_init=w_init )(non_linearity_layer_1) # Shape(1,256,256,1)
	Recon_layer_2 = Elementwise(combine_fn=tf1.add )([Recon_layer_1, hr_image]) # Shape(1,256,256,1)

	return Model(inputs=[lr_image, hr_image], outputs=Recon_layer_2)



def model_D1():	##Phase1 Discriminator

	w_init = tf1.random_normal_initializer(stddev=0.02)
	gamma_init = tf1.random_normal_initializer(1., 0.02)
	df_dim = 64
	lrelu = lambda x: tl.act.lrelu(x, 0.2)

	input_image = Input((None,256,256,1))

	layer_1 = Conv2d( df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(input_image)

	layer_2 = Conv2d( df_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_1)
	layer_2 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_2)

	layer_3 = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_2)
	layer_3 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_3)

	layer_4 = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_3)
	layer_4 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_4)

	layer_5 = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_4)
	layer_5 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_5)

	layer_6 = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_5)
	layer_6 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_6)

	layer_7 = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_6)
	layer_7 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_7)

	layer_8 = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_7)
	layer_8 = BatchNorm2d(gamma_init=gamma_init)(layer_8)

	layer_9 = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_8)
	layer_9 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_9)

	layer_10 = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_9)
	layer_10 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_10)

	layer_11 = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_10)
	layer_11 = BatchNorm2d(gamma_init=gamma_init)(layer_11)

	layer_12 = Elementwise(combine_fn=tf1.add, act=lrelu)([layer_11, layer_8])

	flatten = Flatten()(layer_12)

	dense = Dense(n_units=1, W_init=w_init)(flatten)
	return Model(inputs=input_image, outputs= dense)



def model_D2():	##Phase2 Discriminator

	w_init = tf1.random_normal_initializer(stddev=0.02)
	gamma_init = tf1.random_normal_initializer(1., 0.02)
	df_dim = 64
	lrelu = lambda x: tl.act.lrelu(x, 0.2)

	input_image = Input((None,512,512,3))

	layer_1 = Conv2d( df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(input_image)

	layer_2 = Conv2d( df_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_1)
	layer_2 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_2)

	layer_3 = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_2)
	layer_3 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_3)

	layer_4 = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_3)
	layer_4 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_4)

	layer_5 = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_4)
	layer_5 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_5)

	layer_6 = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(layer_5)
	layer_6 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_6)

	layer_7 = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_6)
	layer_7 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_7)

	layer_8 = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_7)
	layer_8 = BatchNorm2d(gamma_init=gamma_init)(layer_8)

	layer_9 = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_8)
	layer_9 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_9)

	layer_10 = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_9)
	layer_10 = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(layer_10)

	layer_11 = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(layer_10)
	layer_11 = BatchNorm2d(gamma_init=gamma_init)(layer_11)

	layer_12 = Elementwise(combine_fn=tf1.add, act=lrelu)([layer_11, layer_8])

	flatten = Flatten()(layer_12)

	dense = Dense(n_units=1, W_init=w_init)(flatten)
	return Model(inputs=input_image, outputs= dense)


if __name__ == '__main__':
  print(eval('model_'+sys.argv[1]+sys.argv[2]+'()'),"Model Building Successful!")