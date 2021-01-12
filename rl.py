import tensorflow as tf
from tensorflow import keras
import numpy as np


class DDPG(keras.Model):
    def __init__(self, a_dim, s_dim, a_bound, batch_size=32, tau=0.002, gamma=0.8,
                 lr=0.0001, memory_capacity=9000, soft_replace=True):
        super().__init__()
        self.batch_size = batch_size
        self.tau = tau   # soft replacement
        self.gamma = gamma   # reward discount
        self.lr = lr
        self.memory_capacity = memory_capacity

        self.memory = np.zeros((memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self._soft_replace = soft_replace
        self.a_replace_counter = 0
        self.c_replace_counter = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]

        s = keras.Input(shape=(s_dim,))     # current state
        s_ = keras.Input(shape=(s_dim,))    # next state
        self.actor = self._build_actor(s, trainable=True, name="a/eval")
        self.actor_ = self._build_actor(s_, trainable=False, name="a/target")
        self.critic = self._build_critic(s, trainable=True, name="d/eval")
        self.critic_ = self._build_critic(s_, trainable=False, name="d/target")

        self.opt = keras.optimizers.Adam(self.lr, 0.5, 0.9)
        self.mse = keras.losses.MeanSquaredError()

    def _build_actor(self, s, trainable, name):
        x = keras.layers.Dense(self.s_dim * 50, trainable=trainable)(s)
        # x = keras.layers.BatchNormalization(trainable=trainable)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dense(self.s_dim * 50, trainable=trainable)(x)
        # x = keras.layers.BatchNormalization(trainable=trainable)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dense(self.a_dim, trainable=trainable)(x)
        # x = keras.layers.BatchNormalization(trainable=trainable)(x)
        a = self.a_bound * tf.math.tanh(x)
        model = keras.Model(s, a, name=name)
        # model.summary()
        return model

    def _build_critic(self, s, trainable, name):
        a = keras.Input(shape=(self.a_dim,))
        x = tf.concat([
            keras.layers.Dense(self.s_dim * 50, trainable=trainable, activation="relu", use_bias=False)(s),
            keras.layers.Dense(self.a_dim * 50, trainable=trainable, activation="relu", use_bias=False)(a)], axis=1)
        # x = keras.layers.BatchNormalization(trainable=trainable)(x)
        x = keras.layers.Dense(self.s_dim * 50, trainable=trainable)(x)
        # x = keras.layers.BatchNormalization(trainable=trainable)(x)
        x = keras.layers.LeakyReLU()(x)
        q = keras.layers.Dense(1, trainable=trainable)(x)

        model = keras.Model([s, a], q, name=name)
        # model.summary()
        return model

    def param_replace(self):
        if self._soft_replace:
            for la, la_ in zip(self.actor.layers, self.actor_.layers):
                for i in range(len(la.weights)):
                    la_.weights[i] = (1 - self.tau) * la_.weights[i] + self.tau * la.weights[i]
            for lc, lc_ in zip(self.critic.layers, self.critic_.layers):
                for i in range(len(lc.weights)):
                    lc_.weights[i] = (1 - self.tau) * lc_.weights[i] + self.tau * lc.weights[i]
        else:
            self.a_replace_counter += 1
            self.c_replace_counter += 1
            if self.a_replace_counter % 1000 == 0:
                for la, la_ in zip(self.actor.layers, self.actor_.layers):
                    for i in range(len(la.weights)):
                        la_.weights[i] = la.weights[i]
                self.a_replace_counter = 0
            if self.c_replace_counter % 1100 == 0:
                for lc, lc_ in zip(self.critic.layers, self.critic_.layers):
                    for i in range(len(lc.weights)):
                        lc_.weights[i] = lc.weights[i]
                self.c_replace_counter = 0

    def act(self, s):
        if s.ndim < 2:
            s = np.expand_dims(s, axis=0)
        a = self.actor.predict(s)
        return a

    def sample_memory(self):
        if self.memory_full:
            indices = np.random.randint(0, self.memory_capacity, size=self.batch_size)
        else:
            indices = np.random.randint(0, self.pointer, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        return bs, ba, br, bs_

    def learn(self):
        self.param_replace()
        bs, ba, br, bs_ = self.sample_memory()
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic((bs, a))
            actor_loss = tf.reduce_mean(-q)     # maximize q
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            a_ = self.actor_(bs_)
            q_ = br + self.gamma * self.critic_((bs_, a_))
            q = self.critic((bs, ba))
            critic_loss = self.mse(q_, q)
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        return actor_loss.numpy(), critic_loss.numpy()

    def store_transition(self, s, a, r, s_):
        if s.ndim == 1:
            s = np.expand_dims(s, axis=0)
        if s_.ndim == 1:
            s_ = np.expand_dims(s_, axis=0)

        transition = np.concatenate((s, a, np.array([[r]], ), s_), axis=1)
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > self.memory_capacity:      # indicator for learning
            self.memory_full = True
