from __future__ import print_function

import os
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np
from utils_tf import ifft_tf, fft_tf
from utils import *
from SSIM import compute_ssim, PSNR

def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, channel=64, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)   #maximum
    input_im = concat([input_im, input_max])
    with tf.compat.v1.variable_scope('DecomNet', reuse=tf.compat.v1.AUTO_REUSE):
        conv = tf.compat.v1.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")
        for idx in range(layer_num):
            conv = tf.compat.v1.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_%d' % idx)
        conv = tf.compat.v1.layers.conv2d(conv, 4, kernel_size, padding='same', activation=None, name='recon_layer')
        # filter = 4
    R = tf.sigmoid(conv[:,:,:,0:3])   # R o L
    L = tf.sigmoid(conv[:,:,:,3:])

    return R, L

def RelightNet(input_L, input_R, channel=64, kernel_size=3):
    input_im = concat([input_R, input_L])
    with tf.compat.v1.variable_scope('RelightNet'):
        conv0 = tf.compat.v1.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
        conv1 = tf.compat.v1.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv2 = tf.compat.v1.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv3 = tf.compat.v1.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
        up1 = tf.compat.v1.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
        deconv1 = tf.compat.v1.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        up2 = tf.compat.v1.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv2= tf.compat.v1.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
        up3 = tf.compat.v1.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = tf.compat.v1.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
        deconv1_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        feature_fusion = tf.compat.v1.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        output = tf.compat.v1.layers.conv2d(feature_fusion, 1, 3, padding='same', activation=None)
    return output


def my_RelightNet(input_L, channel=64, kernel_size=3):
    # u-net

    with tf.compat.v1.variable_scope('RelightNet'):
        conv0 = tf.compat.v1.layers.conv2d(input_L, channel, kernel_size, padding='same', activation=tf.nn.relu)
        conv1 = tf.compat.v1.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same',
                                           activation=tf.nn.relu)
        conv2 = tf.compat.v1.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same',
                                           activation=tf.nn.relu)
        conv3 = tf.compat.v1.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same',
                                           activation=tf.nn.relu)

        up1 = tf.compat.v1.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
        deconv1 = tf.compat.v1.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        up2 = tf.compat.v1.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv2 = tf.compat.v1.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
        up3 = tf.compat.v1.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = tf.compat.v1.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0

        deconv1_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv1,
                                                                    (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv2,
                                                                    (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        feature_fusion = tf.compat.v1.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        output_mag = tf.compat.v1.layers.conv2d(feature_fusion, 1, 3, padding='same', activation=None)

        output = output_mag
    return output



class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.DecomNet_layer_num = 5

        # build the model
        self.input_low = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)
        [R_high, I_high] = DecomNet(self.input_high, layer_num=self.DecomNet_layer_num)

        # mag_fft_h, ang_fft_h = fft_tf(I_high)
        # mag_fft, ang_fft = fft_tf(I_low)
        # scaler = [tf.reduce_min(mag_fft), tf.reduce_max(mag_fft) - tf.reduce_min(mag_fft)]
        # mag_fft = (mag_fft-scaler[0])/scaler[1]
        # I_fft = my_RelightNet(mag_fft)  #, R_low

        # I_fft = I_fft*scaler[1]+scaler[0]
        # I_delta = ifft_tf(I_fft, mag_fft)
        # print("pred:", I_delta[0])

        I_delta = my_RelightNet(I_low)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])

        self.output_R_low = R_low
        self.output_I_low = I_low_3
        self.output_I_delta = I_delta_3
        self.output_S = R_low * I_delta_3

        # fft loss
        mag_fft_h, ang_fft_h = fft_tf(I_high)
        # mag_fft, ang_fft = fft_tf(I_low)
        mag_fft, ang_fft = fft_tf(I_delta)
        self.mag_loss = tf.reduce_mean(tf.abs(mag_fft_h - mag_fft))
        self.ang_loss = tf.reduce_mean(tf.abs(ang_fft_h - ang_fft))  # ang.


        # Retinex
        ret_h = tf.math.log(R_high) + tf.math.log(I_high_3)
        ret_l = tf.math.log(R_low) + tf.math.log(I_low_3)

        self.retinex_loss = tf.reduce_mean(tf.abs(ret_h - ret_l)) #/ 255 #log
        print("Retinex_loss:", self.retinex_loss)

        # SSIM
        self.recon_ssim_loss = tf.reduce_mean(1 - tf.image.ssim(R_high * I_high_3, R_low * I_low_3, 1.0))

        # loss
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - self.input_low))
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - self.input_high))
        self.recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - self.input_low))
        self.recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - self.input_high))
        self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
        self.relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_3 - self.input_high))

        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        self.loss_Decom = self.recon_loss_low + self.recon_loss_high + 0.001 * self.recon_loss_mutal_low + \
                          0.001 * self.recon_loss_mutal_high + 0.1 * self.Ismooth_loss_low + \
                          0.1 * self.Ismooth_loss_high + 0.01 * self.equal_R_loss# + \
                          # 0.1 * self.recon_ssim_loss
                          #self.retinex_loss

        self.loss_Relight = self.relight_loss + 3 * self.Ismooth_loss_delta #+ \
                            # 0.1 * self.ang_loss #+ self.mag_loss

        self.lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer', epsilon=1e-7) #1e-8

        self.var_Decom = [var for var in tf.compat.v1.trainable_variables() if 'DecomNet' in var.name]
        self.var_Relight = [var for var in tf.compat.v1.trainable_variables() if 'RelightNet' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
        self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list = self.var_Relight)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver_Decom = tf.compat.v1.train.Saver(var_list = self.var_Decom)
        self.saver_Relight = tf.compat.v1.train.Saver(var_list = self.var_Relight)

        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.compat.v1.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data, eval_high_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            output_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
            # R, L
            if train_phase == "Decom":
                # [features, labels]
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low], feed_dict={self.input_low: input_low_eval})

                print("SSIM of Batch:{}".format(compute_ssim(result_1 * result_2, output_high_eval)))

            elif train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta], feed_dict={self.input_low: input_low_eval})


            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)



    def train(self, train_low_data, train_high_data, eval_low_data, eval_high_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom #self.recon_ssim_loss #
            saver = self.saver_Decom
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            saver = self.saver_Relight

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                _, loss = self.sess.run([train_op, train_loss], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, \
                                                                           self.lr: lr[epoch]})

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, eval_high_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0


    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag, model=True):
        tf.compat.v1.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        if model:
            print("[*] Load model from Model")
            load_model_status_Decom, _ = self.load(self.saver_Decom, './model/Decom')
            load_model_status_Relight, _ = self.load(self.saver_Relight, './model/Relight')
        else:
            print("Load model from checkpoint")
            load_model_status_Decom, _ = self.load(self.saver_Decom, './checkpoint/Decom')
            load_model_status_Relight, _ = self.load(self.saver_Relight, './checkpoint/Relight')
        if load_model_status_Decom and load_model_status_Relight:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)

            if input_low_test.shape[-1]==4:   #  (1, 536, 718, 4)
                input_low_test=input_low_test[:,:,:,:3]
                print(input_low_test.shape)

            [R_low, I_low, I_delta, S] = self.sess.run([self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S], feed_dict = {self.input_low: input_low_test})

            if decom_flag == 1:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            save_images(os.path.join(save_dir, name + "_S."   + suffix), S)
            # print("idx:", np.array(test_high_data[idx]).shape, S.shape)  #400, 600, 3) (1, 680, 720, 3)
            print("SSIM of {}: {}".format(name, compute_ssim(S, np.expand_dims(test_high_data[idx],axis=0))))
            print("PSNR of {}: {}".format(name, PSNR(255*np.expand_dims(test_high_data[idx], axis=0), 255*S)))