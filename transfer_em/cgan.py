"""Cycle GAN network.
"""

# Based on tensorflow cycle gan example

# TODO: add support for cloud-based inference

# TODO: add mirrored training strategy
# (this requires normalizing loss based on global batch size, add all variables in mirrored
# scope, and calling training under the mirrored strategy.
# ex: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb)

import tensorflow as tf
import time
from .models.discriminator import *
from .models.generator import *
from .debug import generate_images, accuracy
import numpy as np

class EM2EM(object):
    """Creates CGAN model for 1-channenl 2d or 3d data and provides functions to train and predict.

    Compatible tensor dimension sizes for 2D or 3D are:

    74 
    """

    def __init__(self, dimsize, exp_name, is3d=True, norm_type="instancenorm", ckpt_restore=None, wf=8):
        """Creates model or loads from latest checkpoint for the given exp_name or from the supplied checkpoint.

        Args:
            dimsize (int): dimension size of the input tensor (must be one of the valid sizes)
            exp_name (str): name of the model
            is3d (boolean): true for 3d tensor, otherwise a 2d tensor
            norm_type (str): type of normalization (DEPRECATED -- currently disabled)  
            ckpt_restore (str): path for checkpoint
            wf (int): width factor to decrease the default network width (default=8) (should be 1,2,4,8,16,32)
        """

        if dimsize < 74:
            raise RuntimeError("minimum dimension allowed is 74")

        # enable parallel training
        #self.strategy = tf.distribute.MirroredStrategy()
        #with self.strategy.scope():
        self.discriminator_x = discriminator(is3d, norm_type=norm_type, wf=wf)
        self.discriminator_y = discriminator(is3d, norm_type=norm_type, wf=wf)
        
        self.generator_g, dimsize2 = unet_generator(dimsize, is3d, norm_type=norm_type, wf=wf)
        self.generator_f, _ = unet_generator(dimsize, is3d, norm_type=norm_type, wf=wf)
        # dimsize2 should always be even
        assert((dimsize2 % 2) == 0)
        self.buffer = (dimsize - dimsize2) // 2
        self.outdimsize = dimsize2

        # create optimizers
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.is3d = is3d

        # setup loss functions
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # establish checkpoints
        checkpoint_path = f"./checkpoints/train_{exp_name}"

        self.ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                   generator_f=self.generator_f,
                                   discriminator_x=self.discriminator_x,
                                   discriminator_y=self.discriminator_y,
                                   generator_g_optimizer=self.generator_g_optimizer,
                                   generator_f_optimizer=self.generator_f_optimizer,
                                   discriminator_x_optimizer=self.discriminator_x_optimizer,
                                   discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=50)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_restore is not None:
          self.ckpt.restore(ckpt_restore).assert_existing_objects_matched()
          print (f"checkpoint {ckpt_restore} restored")
        elif self.ckpt_manager.latest_checkpoint:
          self.ckpt.restore(self.ckpt_manager.latest_checkpoint).assert_existing_objects_matched() #.assert_consumed()
          print ('Latest checkpoint restored!!')

    def make_checkpoint(self, epoch_num):
        path = self.ckpt_manager.save()
        print(f"Saving checkpoint for epoch {epoch_num} at {path}")


    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def identity_loss(self, real_image, same_image):
        LAMBDA = 1
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss

    def calc_cycle_loss(self, real_image, cycled_image):
        LAMBDA = 1
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return LAMBDA * loss1


    @tf.function
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)

            crop_func = tf.keras.layers.Cropping2D
            pad_func = tf.keras.layers.ZeroPadding2D
            if self.is3d:
                crop_func = tf.keras.layers.Cropping3D
                pad_func = tf.keras.layers.ZeroPadding3D

            # pad fake data to ensure no off by one issues
            fake_y_pad = pad_func(padding=self.buffer)(fake_y)
            cycled_x = self.generator_f(fake_y_pad, training=True)
            cycled_x_cropped = crop_func(cropping=(self.buffer))(cycled_x)
            # double crop
            real_x_cropped2x = crop_func(cropping=(self.buffer*2))(real_x) 

            fake_x = self.generator_f(real_y, training=True)
            
            # pad fake data to ensure no off by one issues
            fake_x_pad = pad_func(padding=self.buffer)(fake_x)
            cycled_y = self.generator_g(fake_x_pad, training=True)
            cycled_y_cropped = crop_func(cropping=(self.buffer))(cycled_y)
            # double crop
            real_y_cropped2x = crop_func(cropping=(self.buffer*2))(real_y) 

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            # crop real_x
            real_x_cropped = crop_func(cropping=(self.buffer))(real_x) 
            
            same_y = self.generator_g(real_y, training=True)
            # crop real_y
            real_y_cropped = crop_func(cropping=(self.buffer))(real_y) 

            disc_real_x = self.discriminator_x(real_x_cropped, training=True)
            disc_real_y = self.discriminator_y(real_y_cropped, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            # double crop real_y and real_x
            total_cycle_loss = self.calc_cycle_loss(real_x_cropped2x, cycled_x_cropped) + self.calc_cycle_loss(real_y_cropped2x, cycled_y_cropped)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y_cropped, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x_cropped, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)


        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                                self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                                self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                self.discriminator_y.trainable_variables))

        return total_gen_g_loss, total_gen_f_loss, disc_y_loss, disc_x_loss, gen_g_loss, gen_f_loss, total_cycle_loss

    def plot_discriminator(self, location):
        """Plot discriminator.
        """
        tf.keras.utils.plot_model(self.discriminator_x, show_shapes=True, expand_nested=True, to_file=location)

    def plot_generator(self, location):
        """Plot generator.
        """
        tf.keras.utils.plot_model(self.generator_f, show_shapes=True, expand_nested=True, to_file=location)
    
    def train(self, train_input, train_target, epochs=3000, start=0, debug=False, sample=None, sample_gt=None, enable_eager=False, num_samples=4096, check_freq=1):
        """Main function for training model.  This can be run iteratively but re-training will overwrite
        previous saved checkpoints unless 'start' is set.

        If a sample image is provided, an updated image is shown every 100 epochs.
        """

        # enable eager to ease debugging (also could dynamically wrap tf.function(train_step)
        # if only that function needs to be enabled/disabled
        tf.config.experimental_run_functions_eagerly(enable_eager)

        for epoch in range(start, start+epochs):
            start = time.time()
   
            if debug:
                # progress bar
                import tqdm
                with tqdm.tqdm(total=num_samples) as pbar:
                    for data_f, data_g in tf.data.Dataset.zip((train_input, train_target)):
                        loss = self.train_step(data_f, data_g)
                        pbar.update(1)
            else:
                loss = np.zeros((7), dtype=np.float32)
                count = 0
                for data_f, data_g in tf.data.Dataset.zip((train_input, train_target)):
                    loss += self.train_step(data_f, data_g)
                    count += 1
                loss = loss / count
                print(f"Epoch {epoch+1} loss [g_gen_total, f_gen_total, disc_y, disc_x, g_gen_only, f_gen_only, cycle]: {loss}")

            if (epoch + 1) % check_freq == 0:
                self.make_checkpoint(epoch+1)
                # show sample image
                if debug and sample is not None:
                    from IPython.display import clear_output          
                    clear_output(wait=True)
                    sample_pred = self.predict(sample)
                    if sample_gt is not None:
                        crop_func = tf.keras.layers.Cropping2D
                        if self.is3d:
                            crop_func = tf.keras.layers.Cropping3D
                        sample_gt_cropped = crop_func(cropping=(self.buffer))(sample_gt)
                        print(f"Accuracy on sample: {accuracy(sample_gt_cropped[0], sample_pred[0])}")
                    generate_images(sample, sample_pred)
        
            print(f"Time taken for epoch {epoch+1} is {time.time()-start}")

    def predict(self, data):
        """Generate prediction from trained generator.
        """

        return self.generator_g(data)


