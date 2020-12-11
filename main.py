"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from env import ArmEnv
from rl import DDPG
import numpy as np


def set_soft_gpu(yes):
    import tensorflow as tf
    if yes:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def train():
    # start training
    var = 3
    min_var = 0.01
    for ep in range(PARAMS["max_ep"]):
        s = env.reset()
        ep_r = 0.
        for j in range(PARAMS["max_step"]):
            if rl.memory_full:
                env.render()
            a = rl.act(s)
            a = np.clip(np.random.normal(a, var), *env.action_bound)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)
            ep_r += r
            aloss, closs = 0, 0
            if rl.memory_full:
                var = max(var * 0.9998, min_var)
                aloss, closs = rl.learn()

            s = s_
            if done or j == PARAMS["max_step"]-1:
                print('Ep: %d/%d | %s | ep_r: %.1f | var: %.2f | step: %i | aloss=%.4f | closs=%.4f' % (
                    ep, PARAMS["max_ep"], '---' if not done else 'done',
                    ep_r, var, j+1, aloss, closs))
                break
        if rl.memory_full and ep % 20 == 0:
            rl.save_weights("models/{}arms/ep{:03d}".format(PARAMS["n_arms"], ep))


def eval():
    rl.load_weights("models/{}arms/ep{}".format(PARAMS["n_arms"], PARAMS["max_ep"]-1))
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.act(s)
        s, r, done = env.step(a)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--arms", default=2, type=int)
    parser.add_argument("--human", action="store_true", default=False)
    args = parser.parse_args()

    set_soft_gpu(True)
    if args.arms == 2:
        PARAMS = {"training": not args.human, "n_arms": 2, "max_ep": 501, "max_step": 150,
                  "soft_replace": True, "random_target": True, "tau": 0.01, "gamma": 0.9, "lr": 0.0001,
                  "memory_capacity": 9000}
    elif args.arms == 3:
        PARAMS = {"training": not args.human, "n_arms": 3, "max_ep": 1001, "max_step": 200,
                  "soft_replace": True, "random_target": True, "tau": 0.01, "gamma": 0.8, "lr": 0.0001,
                  "memory_capacity": 9000}
    else:
        PARAMS = {"training": not args.human, "n_arms": args.arms, "max_ep": 1001, "max_step": 200,
                  "soft_replace": True, "random_target": True, "tau": 0.01, "gamma": 0.8, "lr": 0.0001,
                  "memory_capacity": 9000}

    # set env
    print(PARAMS)
    env = ArmEnv(n_arms=PARAMS["n_arms"], random_goal=PARAMS["random_target"], on_mouse=False if PARAMS["training"] else True)
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    # set RL method (continuous)
    rl = DDPG(a_dim, s_dim, a_bound, soft_replace=PARAMS["soft_replace"], tau=PARAMS["tau"], gamma=PARAMS["gamma"], lr=PARAMS["lr"], )

    if PARAMS["training"]:
        train()
    else:
        eval()