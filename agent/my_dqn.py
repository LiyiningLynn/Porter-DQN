import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import sys
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Flatten, Lambda, concatenate, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.utils import multi_gpu_model
from keras import backend as K

from agent.util import Memory

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

if not tf.test.gpu_device_name():
    sys.exit()

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

class SimpleDQN:
    def __init__(self, action_space, observation_space, observation_space_local, memory_size,
                 batch_size, learning_rate, gamma, target_update,
                 use_dueling=False, gpu_num=1):
        # parameters
        self.observation_space = observation_space
        self.observation_space_local = observation_space_local
        self.action_space      = action_space #4
        self.memory_size       = memory_size
        self.batch_size        = batch_size
        self.learning_rate     = learning_rate
        self.gamma             = gamma
        self.target_update     = target_update #update target q parameter
        self.use_dueling       = use_dueling
        self.gpu_num           = gpu_num #no


        # variables
        self.train_cnt = 0
        self.memory = Memory(memory_size, observation_space)

        self.target_network = self._get_model()
        self.eval_network   = self._get_model()

    def _get_model(self):
        input_local = Input(shape=self.observation_space_local, dtype='float32')
        input_globa = Input(shape=self.observation_space, dtype='float32')

        #branch local
        xl = Conv2D(32, kernel_size=2, strides=1, padding='same', activation='relu',data_format='channels_first')(input_local)
        xl = Conv2D(32, kernel_size=2, strides=1, padding='same', activation='relu',data_format='channels_first')(xl)
        xl = Flatten()(xl)
        xl = Dense(64, activation='relu')(xl)

        #branch global
        xg = Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu',data_format='channels_first')(input_globa)
        xg = Conv2D(32, kernel_size=2, strides=1, padding='same', activation='relu',data_format='channels_first')(xg)
        xg = Flatten()(xg)
        xg = Dense(64, activation='relu')(xg)

        x = concatenate([xl, xg])

        if self.use_dueling:
            value = Dense(1)(x)
            advantage = Dense(self.action_space, use_bias=False)(x)
            q_vals = Lambda(lambda a: a[0] + (a[1] - K.mean(a[1], keepdims=True)),
                output_shape=(self.action_space,))([value, advantage])
        else:
            q_vals = Dense(self.action_space)(x)

        model = Model(inputs=[input_local, input_globa], outputs=q_vals)
        optimizer = Adam(lr=self.learning_rate)

        if self.gpu_num > 1:
            try:
                model = multi_gpu_model(model, gpus=self.gpu_num)
            except:
                pass

        model.compile(loss=huber_loss, optimizer=optimizer)

        return model
    #obs--a dict containing local and  np.expand_dims(obs, axis=0)
    def get_action(self, obs, eps):
        if eps < np.random.uniform(0, 1):
            q_vals = self.eval_network.predict([np.expand_dims(obs['local'],axis=0),np.expand_dims(obs['global'],axis=0)], batch_size=1)
            action = np.argmax(q_vals[0])
        else:
            action = np.random.randint(self.action_space)

        return action

    def train(self):
        batch_obs, batch_action, batch_reward, batch_nobs = self.memory.sample(self.batch_size)
        batch_target = self._calc_target(batch_obs, batch_action, batch_reward, batch_nobs)
        batch_local = [obs['local'] for obs in batch_obs]
        batch_global= [obs['global'] for obs in batch_obs]
        loss = self.eval_network.train_on_batch([batch_local,batch_global], batch_target)

        self.train_cnt += 1
        if self.train_cnt % self.target_update == 0:
            self._update_target()

        return loss

    def _calc_target(self, batch_obs, batch_action, batch_reward, batch_nobs):
        # TODO: Handle terminal
        n = len(batch_action)
        batch_localn = [obs['local'] for obs in batch_nobs]
        batch_globaln= [obs['global'] for obs in batch_nobs]

        #target_q_vals = self.target_network.predict(batch_nobs, batch_size=self.batch_size)
        #eval_q_vals = self.eval_network.predict(batch_nobs, batch_size=self.batch_size)
        target_q_vals = self.target_network.predict([batch_localn,batch_globaln], batch_size=self.batch_size)
        eval_q_vals = self.eval_network.predict([batch_localn,batch_globaln], batch_size=self.batch_size)
        next_vals = target_q_vals[np.arange(n), np.argmax(eval_q_vals, axis=1)]
        next_vals = batch_reward + self.gamma * next_vals

        batch_local = [obs['local'] for obs in batch_obs]
        batch_global= [obs['global'] for obs in batch_obs]
        targets = self.target_network.predict([batch_local,batch_global], batch_size=self.batch_size)
        targets[np.arange(n), batch_action] = next_vals

        return targets

    def _update_target(self):
        self.target_network.set_weights(self.eval_network.get_weights())

    def save(self, path, episode, i):
        if not os.path.exists(path):
            os.makedirs(path)
        self.target_network.save(path + '/target_network_{:06d}'.format(episode, i) + '.h5')
        self.eval_network.save(path + '/eval_network_{:06d}'.format(episode, i) + '.h5')

    def load(self, path, episode, i):
        self.target_network = load_model(path + '/target_network_{:06d}_{:02d}'.format(episode, i) + '.h5')
        self.eval_network = load_model(path + '/eval_network_{:06d}_{:02d}'.format(episode, i) + '.h5')
