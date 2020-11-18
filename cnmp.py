"""
  @author: yigit.yildirim@boun.edu.tr
"""

import os
import tensorflow as tf
from keras.layers import Input, TimeDistributed, Dense,\
    GlobalAveragePooling1D, Concatenate, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np

import matplotlib.pyplot as plt
import math
import time
# import pylab as pl
import tensorflow_probability as tfp
import keras.losses

# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib
matplotlib.use('Agg')


# This is how code runs on GPU
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Delete above if you want to use GPU

# This is how code runs on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Delete above if you want to use GPU
# data_path = "data/pedsim_"
data_path = "data/1_person_no_obs/"


def dist_generator(d, x, param, noise=0):
    f = (math.exp(-x**2/(2.*param[0]**2))/(math.sqrt(2*math.pi)*param[0]))+param[1]
    return f+(noise*(np.random.rand()-0.5)/100.)


def generate_demonstrations(time_len=200, params=None, title=None):
    fig = plt.figure(figsize=(5, 5))
    x = np.linspace(-0.5, 0.5, time_len)
    times = np.zeros((params.shape[0], time_len, 1))
    times[:] = x.reshape((1, time_len, 1)) + 0.5
    values = np.zeros((params.shape[0], time_len, 1))
    for d in range(params.shape[0]):
            for i in range(time_len):
                values[d, i] = dist_generator(d, x[i], params[d])
            plt.plot(times[d], values[d])
    plt.title(title+' Demonstrations')
    plt.ylabel('Starting Position')
    plt.show()
    return times, values


def sample(data_tuple, num=6):
    to_return = []
    ids = np.random.choice(data_tuple[0].shape[0], num, replace=False)
    for arr in data_tuple:
        to_return.append([arr[ids]])
    for i, lst in enumerate(to_return):  # list to np array
        to_return[i] = np.array(lst[0])
    return tuple(to_return)

#
# X, Y = generate_demonstrations(time_len=200, params=np.array(
#     [[0.6, -0.1], [0.5, -0.23], [0.4, -0.43], [-0.6, 0.1], [-0.5, 0.23], [-0.4, 0.43]]), title='Training')
# v_X, v_Y = generate_demonstrations(time_len=200, params=np.array(
#     [[0.55, -0.155], [0.45, -0.32], [-0.45, 0.32], [-0.55, 0.155]]), title='Validation')
#
# print(f'training X {X.shape}')
# print(f'training Y {Y.shape}')
# print(f'validation X {v_X.shape}')
# print(f'validation Y {v_Y.shape}')
# np.save(data_path + 'training_X', X)
# np.save(data_path + 'training_Y', Y)
# np.save(data_path + 'validation_X', v_X)
# np.save(data_path + 'validation_Y', v_Y)


# X, Y = (np.load(data_path + 'training_X.npy'), np.load(data_path + 'training_Y.npy'))
# v_X, v_Y = (np.load(data_path + 'validation_X.npy'), np.load(data_path + 'validation_Y.npy'))
# X, Y = (np.load(data_path + 'd_x.npy'), np.load(data_path + 'd_y.npy'))
# v_X, v_Y = (np.load(data_path + 'v_d_x.npy'), np.load(data_path + 'v_d_y.npy'))
X, Y = (np.load(data_path + 'd_x.npy'), np.load(data_path + 'd_y.npy'))
v_X, v_Y = (np.load(data_path + 'v_d_x.npy'), np.load(data_path + 'v_d_y.npy'))

(X, Y) = sample((X, Y), num=1)
# (v_X, v_Y) = sample((v_X, v_Y), num=4)
(v_X, v_Y) = (X, Y)

obs_max = 6
d_N = X.shape[0]
d_x, d_y = (X.shape[-1], Y.shape[-1])  # d_x, d_y: dimensions
time_len = X.shape[1]
obs_mlp_layers = [128, 128, 128]
decoder_layers = [128, 128, d_y*2]

print(f'd_N={d_N}')
print(f'obs_max={obs_max}')
print(f'X: {X.shape}, Y: {Y.shape}')
print(f'd_x={d_x}')
print(f'd_y={d_y}')
print(f'time_len={time_len}')


# original get_train_sample
def get_train_sample():
    n = np.random.randint(0, obs_max) + 1
    d = np.random.randint(0, d_N)
    observation = np.zeros((1, n, d_x + d_y))
    target_X = np.zeros((1, 1, d_x))
    target_Y = np.zeros((1, 1, d_y*2))

    perm = np.random.permutation(time_len)
    observation[0, :n, :d_x] = X[d, perm[:n]]
    observation[0, :n, d_x:d_x+d_y] = Y[d, perm[:n]]

    target_X[0, 0] = X[d, perm[n]]
    target_Y[0, 0, :d_y] = Y[d, perm[n]]
    return [observation, target_X], target_Y


def predict_model_(observation, target_X, plot=True):
    predicted_Y = np.zeros((time_len, d_y))
    predicted_std = np.zeros((time_len, d_y))
    prediction = model.predict([observation, target_X])[0]
    predicted_Y = prediction[:, :d_y]
    predicted_std = np.log(1+np.exp(prediction[:, d_y:]))
    if plot:  # We highly recommend that you customize your own plot function, but you can use this function as default
        for i in range(d_y):  # for every feature in Y vector we are plotting training data and its prediction
            fig = plt.figure(figsize=(5, 5))
            for j in range(d_N):
                plt.plot(X[j, :, 0], Y[j, :, i])  # assuming X[j,:,0] is time
            plt.plot(X[j, :, 0], predicted_Y[:, i], color='black')
            plt.errorbar(X[j, :, 0], predicted_Y[:, i], yerr=predicted_std[:, i], color='black', alpha=0.4)
            plt.scatter(observation[0, :, 0], observation[0, :, d_x+i], marker="X", color='black')
            plt.show()
    return predicted_Y, predicted_std


def predict_model(observation, target_X, plot=True, final=False):  # observation and target_X contain gamma values
    predicted_Y = np.zeros((time_len, d_y))
    predicted_std = np.zeros((time_len, d_y))
    prediction = model.predict([observation, target_X])[0]
    predicted_Y = prediction[:, :d_y]
    predicted_std = np.log(1+np.exp(prediction[:, d_y:]))
    if plot:  # We highly recommend that you customize your own plot function, but you can use this function as default
        if final:
            for i in range(d_y):  # for every feature in Y vector we are plotting training data and its prediction
                fig = plt.figure(figsize=(5, 5))
                for j in range(d_N):
                    plt.plot(X[j, :, 0][::-1], Y[j, :, i])  # assuming X[j,:,0] is time
                plt.plot(X[j, :, 0][::-1], predicted_Y[:, i], color='black')
                plt.errorbar(X[j, :, 0][::-1], predicted_Y[:, i], yerr=predicted_std[:, i], color='black', alpha=0.4)
                plt.scatter(observation[0, :, 0], observation[0, :, d_x+i], marker="X", color='black')
                plt.savefig(f'output/model_pred_{str(int(time.time()))}_dim_{i}_(final)')
                plt.close()
        else:
            for i in range(d_y):  # for every feature in Y vector we are plotting training data and its prediction
                fig = plt.figure(figsize=(5, 5))
                for j in range(d_N):
                    plt.plot(X[j, :, 0][::-1], Y[j, :, i])  # assuming X[j,:,0] is time
                plt.plot(X[j, :, 0][::-1], predicted_Y[:, i], color='black')
                plt.errorbar(X[j, :, 0][::-1], predicted_Y[:, i], yerr=predicted_std[:, i], color='black', alpha=0.4)
                plt.scatter(observation[0, :, -1], observation[0, :, d_x+i], marker="X", color='black')
                plt.savefig(f'output/model_pred_{str(int(time.time()))}_dim_{i}')
                plt.close()
    return predicted_Y, predicted_std


def custom_loss(y_true, y_predicted):
    mean, log_sigma = tf.split(y_predicted, 2, axis=-1)
    y_true_value, temp = tf.split(y_true, 2, axis=-1)
    sigma = tf.nn.softplus(log_sigma)
    dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
    loss = -tf.reduce_mean(dist.log_prob(y_true_value))
    return loss


def MLP(input_dim, layers, name="mlp", parallel_inputs=False):
    input_layer = Input(shape=(None, input_dim), name=name+'_input')
    for i in range(len(layers)-1):
        hidden = TimeDistributed(Dense(layers[i], activation='relu'), name=name+'_'+str(i))\
            (input_layer if i == 0 else hidden) if parallel_inputs else \
            Dense(layers[i], activation='relu', name=name+'_'+str(i))(input_layer if i == 0 else hidden)

    hidden = TimeDistributed(Dense(layers[-1]), name=name+'_output')(hidden) \
        if parallel_inputs else Dense(layers[-1], name=name+'_output')(hidden)

    return Model(input_layer, hidden, name=name)


observation_layer = Input(shape=(None, d_x+d_y), name="observation")  # (x_o,y_o) tuples
target_X_layer = Input(shape=(None, d_x), name="target")  # x_q

ObsMLP = MLP(d_x+d_y, obs_mlp_layers, name='obs_mlp', parallel_inputs=True)  # Network E
obs_representations = ObsMLP(observation_layer)  # r_i
general_representation = GlobalAveragePooling1D()(obs_representations)  # r
general_representation = Lambda(lambda x: tf.keras.backend.repeat(x[0], tf.shape(x[1])[1]), name='Repeat')\
    ([general_representation, target_X_layer])  # r in batch form (same)

merged_layer = Concatenate(axis=2, name='merged')([general_representation, target_X_layer])  # (r,x_q) tuple
Decoder = MLP(d_x+obs_mlp_layers[-1], decoder_layers, name='decoder_mlp', parallel_inputs=False)  # Network Q
output = Decoder(merged_layer)  # (mean_q, std_q)

model = Model([observation_layer, target_X_layer], output)
model.compile(optimizer=Adam(lr=1e-4), loss=custom_loss)
model.summary()

plot_model(model, to_file='model.png')


def generator():
    while True:
        inp, out = get_train_sample()
        yield (inp, out)


class CNMP_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.smooth_losses = [0]
        self.losses = []
        self.step = 0
        self.loss_checkpoint = 5000
        self.plot_checkpoint = 100000
        self.validation_checkpoint = 100
        self.validation_error = 9999999
        return

    # original on_batch_end
    def on_batch_end(self, batch, logs={}):
        if self.step % self.validation_checkpoint == 0:
            # Here, you should customize our own validation function according to your data and save your best model
            current_error = 0
            for i in range(v_X.shape[0]):
                # predicting whole trajectory by using the first time step of the ith validation
                # trajectory as given observation
                predicted_Y, predicted_std = predict_model(np.concatenate((v_X[i, 0], v_Y[i, 0])).reshape(1, 1, d_x+d_y)
                                                           , v_X[i].reshape(1, time_len, d_x), plot=False)
                current_error += np.mean((predicted_Y - v_Y[i, :])**2) / v_X.shape[0]
            if current_error < self.validation_error:
                self.validation_error = current_error
                model.save('cnmp_best_validation.h5')
                print(f' New validation best. Error is {current_error}')
            # If you are not using validation, please note that every large-enough nn model will eventually
            # overfit to the input data

        if self.step % self.loss_checkpoint == 0:
            self.losses.append(logs.get('loss'))
            self.smooth_losses[-1] += logs.get('loss')/(self.plot_checkpoint/self.loss_checkpoint)

        if self.step % self.plot_checkpoint == 0:
            print(self.step)
            # clearing output cell
            # display.clear_output(wait=True)
            # display.display(pl.gcf())

            # plotting training and smoothed losses
            plt.figure(figsize=(15, 5))
            plt.subplot(121)
            plt.title('Train Loss')
            plt.plot(range(len(self.losses)), self.losses)
            plt.subplot(122)
            plt.title('Train Loss (Smoothed)')
            plt.plot(range(len(self.smooth_losses)), self.smooth_losses)
            # plt.show()
            plt.savefig("output/" + str(self.step))
            plt.close()

            # plotting on-train examples by user given observations
            for i in range(v_X.shape[0]):
                # for each validation trajectory, predicting and plotting whole trajectories by using
                # the first time steps as given observations.
                predict_model(np.concatenate((v_X[i, 0], v_Y[i, 0])).
                              reshape(1, 1, d_x+d_y), v_X[i].reshape(1, time_len, d_x))

            if self.step != 0:
                self.smooth_losses.append(0)

        self.step += 1
        return


max_training_step = 1000000
model.fit_generator(generator(), steps_per_epoch=max_training_step, epochs=1, verbose=1, callbacks=[CNMP_Callback()])

keras.losses.custom_loss = custom_loss
model = load_model('cnmp_best_validation.h5', custom_objects={'tf': tf})

predicted_Y, predicted_std = predict_model(np.concatenate(([[12.5], [0]], [[1.25], [0.02]])).
                                           reshape(1, -1, d_x+d_y), X[0].reshape(1, time_len, d_x), final=True)

# predicted_Y, predicted_std = predict_final(int(time_len/2), X[0].reshape(1, time_len, d_x))

# predicted_Y, predicted_std = predict_model(np.concatenate(([[0.0]], [[1.75]]), axis=1).reshape(1, -1, d_x+d_y), X[0].
#                                            reshape(1, time_len, d_x))
#
# predicted_Y, predicted_std = predict_model(np.concatenate(([[0.5]], [[2.8]]), axis=1).reshape(1, -1, d_x+d_y), X[0].
#                                            reshape(1, time_len, d_x))
