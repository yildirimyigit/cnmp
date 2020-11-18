"""
  @author: yigit.yildirim@boun.edu.tr
"""

import os
from keras.models import Model, load_model
import keras.losses
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

import time

import matplotlib
matplotlib.use('Agg')


def custom_loss(y_true, y_predicted):
    mean, log_sigma = tf.split(y_predicted, 2, axis=-1)
    y_true_value, temp = tf.split(y_true, 2, axis=-1)
    sigma = tf.nn.softplus(log_sigma)
    dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
    loss = -tf.reduce_mean(dist.log_prob(y_true_value))
    return loss


root_path = f'output/sfm/1_obs_huge/'
model_path = f'{root_path}1604977395/'
# output_path = f'{model_path}prediction_{str(int(time.time()))}/'
# os.mkdir(output_path)
data_root = "/home/yigit/phd/yigit_phd_thesis/cnmp/data/sfm/1_obs_huge/"
data_path = f"{data_root}demonstrations/"

# keras.losses.custom_loss = custom_loss
# model = load_model(f'{model_path}cnmp_best_validation.h5', custom_objects={'tf': tf})
#
# latent_layer = Model(inputs=model.input, outputs=model.get_layer('obs_mlp').get_output_at(-1))

# observation = np.array([0.0831583e-02, 2.8e+01, 1.006e+00, 14.0, 0.0, 2.01129232e-04]).reshape(1, 1, 6)
# target_X_Gamma = np.array([0, 27, 1, 13]).reshape(1, 1, 4)
# a = latent_layer.predict([observation, target_X_Gamma])
#
# print(a.shape)  # (1, 1, 256)


X, Y, gamma = (np.load(data_path + 'd_x.npy'), np.load(data_path + 'd_y.npy'), np.load(data_path + 'd_gamma.npy'))

print('test')
