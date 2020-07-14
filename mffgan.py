from utils import  get_train_dataset, generate_bicubic_samples, PSNR, SSIM, cast_uint8
from model_list import get_model
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import time
import cv2 as cv
import numpy as np
import tensorlayer as tl
import tensorflow as tf2
import tensorflow.compat.v1 as tf1
from tensorlayer.models import Model

class MIFF:
	
	def __init__(self, CONFIG):

		if CONFIG.mode==1 :
		
			gen_model = get_model('G', CONFIG.gen_model)
			dis_model = get_model('D', CONFIG.dis_model)
			VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

			lr_init = 1e-4
			lr_v = tf1.Variable(lr_init)
			beta1 = 0.9
			n_epoch_init = CONFIG.init_epoch # n_epoch_init =20 # 
			n_epoch = CONFIG.total_epoch # n_epoch = 100 # 
			batch_size = CONFIG.batch_size # batch_size = 8 # 
			decay_every = int(n_epoch/ 2)
			lr_decay = 0.1
			resume_epoch = 0
			stats=[['MPSNR','BPSNR','MSSIM','BSSIM','Model','EPOCH']]
			bi_psnr = []
			mo_psnr = []
			bi_ssim = []
			mo_ssim = []
			if CONFIG.load_weights:
			
				resume_epoch = CONFIG.model_epoch
			
				if CONFIG.gan_init:
					gen_model.load_weights('Checkpoints_MIFFGAN/MIFFGAN_INIT_{}_EPID_{}.h5'.format(CONFIG.gen_model, CONFIG.model_epoch))
					resume_epoch = 0
				else:	
					gen_model.load_weights('Checkpoints_MIFFGAN/MIFFGAN_{}_EPID_{}.h5'.format(CONFIG.gen_model, CONFIG.model_epoch))
					dis_model.load_weights('Checkpoints_MIFFGAN/MIFFDIS_{}_GAN_{}_EPID_{}.h5'.format(CONFIG.dis_model, CONFIG.gen_model, CONFIG.model_epoch))
				
		
			g_optimizer_init = tf2.optimizers.Adam(lr_v, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
			g_optimizer = tf2.optimizers.Adam(lr_v, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
			d_optimizer = tf2.optimizers.Adam(lr_v, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

			gen_model.train()
			dis_model.train()
			VGG.train()

			train_ds = get_train_dataset(CONFIG)

			if not CONFIG.load_weights or CONFIG.gan_init:
			
				print('##	initial learning (G)')
				
				for epoch in range(n_epoch_init):
					for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
						if lr_patchs.shape[0] != batch_size:  
							break
						step_time = time.time()
						with tf1.GradientTape() as tape:
							out_bicu = generate_bicubic_samples(lr_patchs.numpy())
							fake_patchs = gen_model([lr_patchs, out_bicu])
							mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
						grad = tape.gradient(mse_loss, gen_model.trainable_weights)
						g_optimizer_init.apply_gradients(zip(grad, gen_model.trainable_weights))
						print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.6f} ".format(
						epoch+1+resume_epoch, resume_epoch+n_epoch_init, step+1, CONFIG.no_of_batches, time.time() - step_time, mse_loss))
					
					path = 'Training_MIFFGAN/gan_init_{}_train_{}.png'.format(CONFIG.gen_model, epoch+1+resume_epoch)
					tl.vis.save_images(cast_uint8(fake_patchs.numpy()), [2, CONFIG.batch_size//2], path)
					
					if ((epoch+1+resume_epoch) % CONFIG.save_interval) == 0:
						gen_model.save_weights('Checkpoints_MIFFGAN/MIFFGAN_INIT_{}_EPID_{}.h5'.format(CONFIG.gen_model, epoch+1+resume_epoch))
				
				gen_model.save_weights('Checkpoints_MIFFGAN/MIFFGAN_INIT_{}_EPID_{}.h5'.format(CONFIG.gen_model, n_epoch_init + resume_epoch))
			
			
			
			print('##	adversarial learning (G, D)')
				
			for epoch in range(n_epoch):

				for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
					if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
						break
					step_time = time.time()
					with tf1.GradientTape(persistent=True) as tape:
		      
						out_bicu = generate_bicubic_samples(np.squeeze(lr_patchs,axis=3))
						fake_patchs = gen_model([lr_patchs, out_bicu])

						logits_fake = dis_model(fake_patchs)
						logits_real = dis_model(hr_patchs)

						feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
						feature_real = VGG((hr_patchs+1)/2.)
		      
						d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf1.ones_like(logits_real))
						d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf1.zeros_like(logits_fake))
						d_loss = d_loss1 + d_loss2
			      
						g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf1.ones_like(logits_fake))
						mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)

						vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
						g_loss = mse_loss + vgg_loss + g_gan_loss

					grad = tape.gradient(g_loss, gen_model.trainable_weights)
					g_optimizer.apply_gradients(zip(grad, gen_model.trainable_weights))
					grad = tape.gradient(d_loss, dis_model.trainable_weights)
					d_optimizer.apply_gradients(zip(grad, dis_model.trainable_weights))
					print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.6f}, vgg:{:.6f}, adv:{:.6f}) d_loss: {:.6f}".format(
					epoch+1+resume_epoch, resume_epoch + n_epoch, step+1, CONFIG.no_of_batches, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))
					
				# update the learning rate
				if (epoch+resume_epoch) % decay_every == 0:
					new_lr_decay = lr_decay**((epoch+resume_epoch)// decay_every)
					lr_v.assign(lr_init * new_lr_decay)
					log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
					print(log)

				if (epoch+1+resume_epoch)%  CONFIG.save_interval == 0:
					gen_model.save_weights('Checkpoints_MIFFGAN/MIFFGAN_{}_EPID_{}.h5'.format(CONFIG.gen_model, epoch+1+resume_epoch))
					dis_model.save_weights('Checkpoints_MIFFGAN/MIFFDIS_{}_GAN_{}_EPID_{}.h5'.format(CONFIG.dis_model, CONFIG.gen_model, epoch+1+resume_epoch))
					print("Save time: {}".format(time.asctime( time.localtime(time.time()))))
					for i in range(CONFIG.batch_size):
						lrimg = np.squeeze(lr_patchs[i], axis =-1)
						lrimg = np.pad(lrimg, ((64, 64), (64, 64)), constant_values=(255.0))
						opimg = cast_uint8(fake_patchs[i].numpy())
						combine_imgs = np.concatenate((lrimg[:,:,np.newaxis], out_bicu[i], opimg, hr_patchs[i]), axis = 1)
						path = 'Training_MIFFGAN/id_{}_gan_{}_train_{}.png'.format(i+1, CONFIG.gen_model, epoch+1+resume_epoch)
						tl.vis.save_image(combine_imgs,path)
						path = 'Training_MIFFGAN/gn_{}_ep_{}_id_{}_'.format(CONFIG.gen_model, epoch+1+resume_epoch, i+1)
						tl.vis.save_image(lrimg[:,:,np.newaxis],path+'lr.png')
						tl.vis.save_image(hr_patchs[i],path+'hr.png')
						tl.vis.save_image(out_bicu[i],path+'bc.png')
						r= np.amin(lrimg)-np.amin(opimg)
						np.add(opimg,r)
						tl.vis.save_image(opimg, path+'op.png')

				  

		elif CONFIG.mode==2:	## Validation

			model = get_model('G', CONFIG.gen_model)
			model.load_weights('Checkpoints_MIFFGAN/MIFFGAN_{}_EPID_{}.h5'.format(CONFIG.gen_model, CONFIG.model_epoch))
			model.eval()  ## disable dropout, batch norm moving avg ...

			save_time = time.time()
			
			## Reading Valiiation dataset
			lrimg_file_list = tl.files.load_file_list(path=CONFIG.dir_val_in, regx='.*.png', printable=False)
			hrimg_file_list = tl.files.load_file_list(path=CONFIG.dir_val_target, regx='.*.png', printable=False)
			lrimg_file_list.sort(key=tl.files.natural_keys)
			hrimg_file_list.sort(key=tl.files.natural_keys)
			lrimg_list = np.array(tl.vis.read_images(lrimg_file_list, path=CONFIG.dir_val_in, n_threads=32))
			hrimg_list = np.array(tl.vis.read_images(hrimg_file_list, path=CONFIG.dir_val_target, n_threads=32)) 
			
			
			lrimg_list = lrimg_list[:,:,:,np.newaxis]
			hrimg_list = hrimg_list[:,:,:,np.newaxis]

			bcimg_list = generate_bicubic_samples(lrimg_list)
			opimg_list = model([tf1.cast(lrimg_list,tf1.float32), tf1.cast(bcimg_list,tf1.float32)]) 
			opimg_list = opimg_list.numpy()

			bicubic_psnr, model_psnr = PSNR (hrimg_list , bcimg_list, opimg_list)
			bicubic_ssim, model_ssim = SSIM (hrimg_list , bcimg_list, opimg_list)
			
			for i in range(lrimg_list.shape[0]):
				name= lrimg_file_list[i].split('/')[-1].split('.')[0]
				lrimg = np.pad(lrimg_list[i], ((64, 64), (64, 64),(0,0)), constant_values=(255.0))

				combine_imgs= np.concatenate((lrimg[:,:,np.newaxis], bcimg_list[i], opimg_list[i], hrimg_list[i]), axis = 1)
				path = 'Validation/Val_MIFFGAN/{}_gan_{}_val_{}.png'.format(name, CONFIG.gen_model, CONFIG.model_epoch)
				tl.vis.save_images(combine_imgs, path)
            
			print(np.stack((model_psnr, bicubic_psnr), axis=-1))
			print(np.stack((model_ssim, bicubic_ssim), axis=-1))
			print(np.subtract(model_psnr, bicubic_psnr))
			print('SUM(PSNR DIFF): {}'.format(np.sum(np.subtract(model_psnr, bicubic_psnr))))
			print('AVG MODEL PSNR: {}, AVG BICUBIC PSNR: {}'.format(np.sum(model_psnr)/lrimg_list.shape[0], np.sum(bicubic_psnr)/lrimg_list.shape[0]))
			print('SUM(SSIM DIFF): {}'.format(np.sum(np.subtract(model_ssim, bicubic_ssim))))
			print('AVG MODEL SSIM: {}, AVG BICUBIC SSIM: {}'.format(np.sum(model_ssim)/lrimg_list.shape[0], np.sum(bicubic_ssim)/lrimg_list.shape[0]))
			print((time.time()-save_time)/10)
			
			 
			 



		 
