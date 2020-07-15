from mffgan import mffgan
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorlayer as tl
#import tensorflow as tf2 
import tensorflow.compat.v1 as tf1

flags = tf1.app.flags

flags.DEFINE_integer('mode', 1, '1 for Training/ 2 for Validation')
flags.DEFINE_boolean('load_weights', False, 'Resume training with load weights')
flags.DEFINE_boolean('gan_init', False, 'GAN training with/without initial training')
flags.DEFINE_integer('gen_model', 0, 'Generator model number')
flags.DEFINE_integer('dis_model', 0, 'Discreminator model number')
flags.DEFINE_integer('model_epoch', 0, 'Model epoch number')
flags.DEFINE_integer('init_epoch', 0, 'Initial training epoch')
flags.DEFINE_integer('total_epoch', 0, 'Total training epoch')
flags.DEFINE_integer('batch_size', 0, 'Batch_size')
flags.DEFINE_integer('no_of_batches', 0, 'Number of batches')
flags.DEFINE_integer('save_interval', 0, 'Save model interval')
flags.DEFINE_integer('size', 128, 'LR size')
flags.DEFINE_integer('scale', 4, 'scaling factor')
flags.DEFINE_string('dir_train_in', 'Training_inputs/SR_target', 'Input folder for training')
flags.DEFINE_string('dir_val_in', 'Validation_inputs/Test', 'Input folder for Validation')
flags.DEFINE_string('dir_val_target', 'Validation_inputs/Target', 'Target folder for Validation')
CONFIG=flags.FLAGS

def main(args):
		mffgan(CONFIG)
if __name__ == '__main__':
    tf1.app.run()
