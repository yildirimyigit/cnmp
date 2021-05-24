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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Delete above if you want to use GPU

# This is how code runs on CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Delete above if you want to use GPU
# data_path = "data/pedsim_"
data_path = "data/sfm/continuous_poses_1/new/combined/demonstrations/"
novel_data_path = "data/sfm/continuous_poses_1/new/combined/demonstrations/"

output_root_path = f'output/sfm/continuous_poses_1/new/combined/'
output_path = f'{output_root_path}{str(int(time.time()))}/'
model_preds_path = f'{output_path}model_preds/'

try:
    os.mkdir(output_root_path)
except:
    pass
try:
    os.mkdir(output_path)
except:
    pass
try:
    os.mkdir(model_preds_path)
except:
    pass


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
    for i, lst in enumerate(to_return):
        to_return[i] = np.array(lst[0])
    return tuple(to_return)


# X, Y = (np.load(data_path + 'd_x.npy'), np.load(data_path + 'd_y.npy'))
# v_X, v_Y = (np.load(data_path + 'v_d_x.npy'), np.load(data_path + 'v_d_y.npy'))
X, Y, gamma = (np.load(data_path + 'd_x.npy'), np.load(data_path + 'd_y.npy'), np.load(data_path + 'd_gamma.npy'))
v_X, v_Y, v_gamma = (np.load(data_path + 'v_d_x.npy'), np.load(data_path + 'v_d_y.npy'),
                     np.load(data_path + 'v_d_gamma.npy'))

(X, Y, gamma) = sample((X, Y, gamma), num=4750)
(v_X, v_Y, v_gamma) = sample((v_X, v_Y, v_gamma), num=502)
(novel_X, novel_Y, novel_gamma) = v_X[-2:], v_Y[-2:], v_gamma[-2:]
(v_X, v_Y, v_gamma) = v_X[:-2], v_Y[:-2], v_gamma[:-2]
# (X, Y, gamma) = sample((X, Y, gamma), num=1)
# v_X, v_Y, v_gamma = np.copy(X), np.copy(Y), np.copy(gamma)

# ###############################################
# novel trajectory to be used in the final prediction
# novel_X, novel_Y, novel_gamma = (np.load(novel_data_path + 'd_x.npy'), np.load(novel_data_path + 'd_y.npy'),
#                                  np.load(novel_data_path + 'd_gamma.npy'))

# (novel_X, novel_Y, novel_gamma) = sample((novel_X, novel_Y, novel_gamma), num=1)  # The actual line
# (novel_X, novel_Y, novel_gamma) = v_X[-1], v_Y[-1], v_gamma[-1]
# ###############################################

obs_max = 10
d_N = X.shape[0]
d_x, d_y, d_gamma = (X.shape[-1], Y.shape[-1], gamma.shape[-1])  # d_x, d_y: dimensions
time_len = X.shape[1]
obs_mlp_layers = [128, 384, 128]
decoder_layers = [128, 256, 128, d_y*2]

print(f'd_N={d_N}')
print(f'obs_max={obs_max}')
print(f'X: {X.shape}, Y: {Y.shape}, gamma:{gamma.shape}')
print(f'd_x={d_x}')
print(f'd_y={d_y}')
print(f'd_gamma={d_gamma}')
print(f'time_len={time_len}')


# original get_train_sample
def get_train_sample_():
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


def get_train_sample():
    n = np.random.randint(0, obs_max) + 1
    d = np.random.randint(0, d_N)
    observation = np.zeros((1, n, d_x + d_y + d_gamma))
    target_X_gamma = np.zeros((1, 1, d_x + d_gamma))
    target_Y = np.zeros((1, 1, d_y*2))  # d_y*2: 1 for mean and 1 for variance for any d_y dimension

    # Following lines are just for randomly selecting n+1 points on the trajectory? (+1 is just for the target)
    perm = np.random.permutation(time_len)
    observation[0, :n, :d_x] = X[d, perm[:n]]
    observation[0, :n, d_x:d_x+d_gamma] = gamma[d, perm[:n]]  # gamma concatenation to observation
    observation[0, :n, d_x+d_gamma:d_x+d_gamma+d_y] = Y[d, perm[:n]]

    target_X_gamma[0, 0, :d_x] = X[d, perm[n]]
    target_X_gamma[0, 0, d_x:d_x+d_gamma] = gamma[d, perm[n]]  # gamma concatenation to target query
    target_Y[0, 0, :d_y] = Y[d, perm[n]]
    return [observation, target_X_gamma], target_Y


def predict_model(observation, target_X, plot=True, final=False):  # observation and target_X contain gamma values
    predicted_Y = np.zeros((time_len, d_y))
    predicted_std = np.zeros((time_len, d_y))
    prediction = model.predict([observation, target_X])[0]
    predicted_Y = prediction[:, :d_y]
    predicted_std = np.log(1+np.exp(prediction[:, d_y:]))

    time_range = range(time_len)

    if plot:  # We highly recommend that you customize your own plot function, but you can use this function as default
        if final:
            for i in range(d_y):  # for every feature in Y vector we are plotting training data and its prediction
                fig = plt.figure(figsize=(5, 5))
                # for j in range(d_N):
                #     plt.plot(time_range, Y[j, :, i])
                plt.plot(time_range, novel_Y[0, :, i], color='red')
                plt.plot(time_range, predicted_Y[:, i], color='black')
                plt.errorbar(time_range, predicted_Y[:, i], yerr=predicted_std[:, i], color='black', alpha=0.4)
                # plt.scatter(observation[0, 0, 0], observation[0, :, d_x+d_gamma+i], marker="X", color='black')
                plt.savefig(f'{model_preds_path}model_pred_{str(int(time.time()))}_dim_{i}_(final)')
                plt.close()
        else:
            for i in range(d_y):  # for every feature in Y vector we are plotting training data and its prediction
                fig = plt.figure(figsize=(5, 5))
                for j in range(d_N):
                    plt.plot(time_range, Y[j, :, i])
                plt.plot(time_range, predicted_Y[:, i], color='black')
                plt.errorbar(time_range, predicted_Y[:, i], yerr=predicted_std[:, i], color='black', alpha=0.4)
                plt.scatter(observation[0, :, -1], observation[0, :, d_x+d_gamma+i], marker="X", color='black')
                plt.savefig(f'{model_preds_path}model_pred_{str(int(time.time()))}_dim_{i}')
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


observation_layer = Input(shape=(None, d_x+d_gamma+d_y), name="observation")  # (x_o,y_o) tuples
target_X_layer = Input(shape=(None, d_x+d_gamma), name="target")  # x_q

ObsMLP = MLP(d_x+d_gamma+d_y, obs_mlp_layers, name='obs_mlp', parallel_inputs=True)  # Network E
obs_representations = ObsMLP(observation_layer)  # r_i
general_representation = GlobalAveragePooling1D()(obs_representations)  # r
general_representation = Lambda(lambda x: tf.keras.backend.repeat(x[0], tf.shape(x[1])[1]), name='Repeat')\
    ([general_representation, target_X_layer])  # r in batch form (same)

merged_layer = Concatenate(axis=2, name='merged')([general_representation, target_X_layer])  # (r,x_q) tuple
Decoder = MLP(d_x+d_gamma+obs_mlp_layers[-1], decoder_layers, name='decoder_mlp', parallel_inputs=False)  # Network Q
output = Decoder(merged_layer)  # (mean_q, std_q)

model = Model([observation_layer, target_X_layer], output)
model.compile(optimizer=Adam(lr=1e-4), loss=custom_loss)
model.summary()

plot_model(model, to_file=f'{output_path}model.png')


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
    def on_batch_end_(self, batch, logs={}):
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

    def on_batch_end(self, batch, logs={}):
        if self.step % self.validation_checkpoint == 0:
            # Here, you should customize our own validation function according to your data and save your best model
            current_error = 0
            for i in range(v_X.shape[0]):
                # predicting whole trajectory by using the first time step of the ith validation
                # trajectory as given observation
                observation = np.concatenate((v_X[i, 0], v_gamma[i, 0], v_Y[i, 0])).reshape(1, 1, d_x+d_gamma+d_y)
                target_X_gamma = np.concatenate((v_X[i].reshape(1, time_len, d_x),
                                                 v_gamma[i].reshape(1, time_len, d_gamma)), axis=2)
                predicted_Y, predicted_std = predict_model(observation, target_X_gamma, plot=False)
                current_error += np.mean((predicted_Y - v_Y[i, :])**2) / v_X.shape[0]
            if current_error < self.validation_error:
                self.validation_error = current_error
                model.save(f'{output_path}cnmp_best_validation.h5')
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
            plt.savefig(f'{output_path}{str(self.step)}')
            plt.close()

            # plotting on-train examples by user given observations
            for i in range(v_X.shape[0]):
                # for each validation trajectory, predicting and plotting whole trajectories by using
                # the first time steps as given observations.
                observation = np.concatenate((v_X[i, 0], v_gamma[i, 0], v_Y[i, 0])).reshape(1, 1, d_x + d_gamma + d_y)
                target_X_gamma = np.concatenate((v_X[i].reshape(1, time_len, d_x),
                                                 v_gamma[i].reshape(1, time_len, d_gamma)), axis=2)

                plotting = False
                if np.random.uniform() < 0.05:
                    plotting = True

                predict_model(observation, target_X_gamma, plot=plotting)

            if self.step != 0:
                self.smooth_losses.append(0)

        self.step += 1
        return


max_training_step = 1000000
model.fit_generator(generator(), steps_per_epoch=max_training_step, epochs=1, verbose=1, callbacks=[CNMP_Callback()])

keras.losses.custom_loss = custom_loss
model = load_model(f'{output_path}cnmp_best_validation.h5', custom_objects={'tf': tf, 'custom_loss': custom_loss})

conditioning_step = np.random.choice(novel_X.shape[1], 1)
print(f'Conditioning on the step {conditioning_step} - X: {novel_X[0, conditioning_step]}, '
      f'Y: {novel_Y[0, conditioning_step]}, G: {novel_gamma[0, conditioning_step]}')

observation = np.concatenate((novel_X[0, conditioning_step], novel_gamma[0, conditioning_step],
                              novel_Y[0, conditioning_step])).reshape(1, 1, d_x+d_gamma+d_y)
target_X_gamma = np.concatenate((novel_X[0].reshape(1, time_len, d_x), novel_gamma[0].reshape(1, time_len, d_gamma)),
                                axis=2)

predicted_Y, predicted_std = predict_model(observation, target_X_gamma, final=True)

# conditioning_step = 200
# observation = np.concatenate((v_X[0, conditioning_step], v_gamma[0, conditioning_step],
#                               v_Y[0, conditioning_step])).reshape(1, 1, d_x+d_gamma+d_y)
#
# target_X_gamma = np.concatenate((v_X[0].reshape(1, time_len, d_x), v_gamma[0].reshape(1, time_len, d_gamma)),
#                                 axis=2)

# predicted_Y, predicted_std = predict_model(np.concatenate(([[12.5], [0]], [[1.25], [0.02]])).
#                                            reshape(1, -1, d_x+d_y), X[0].reshape(1, time_len, d_x), final=True)

# predicted_Y, predicted_std = predict_model(observation, target_X_gamma, final=True)

# np.save(f'{output_path}predicted_velocities.npy', predicted_Y)
