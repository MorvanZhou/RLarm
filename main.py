"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
import os
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--arms", default=2, type=int)
parser.add_argument("-fps", "--fps", type=int, default=-1)
parser.add_argument("--show_fps", action="store_true", default=False)
parser.add_argument("--hide", action="store_true", default=False)
parser.add_argument("-l", "--load", default=0, type=int)
args = parser.parse_args()

if args.hide:
    os.environ["DISPLAY"] = ":1"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # if only use cpu

from env import ArmEnv
from rl import DDPG


def train():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    # start training
    var = 2
    min_var = 0.01
    for ep in range(PARAMS["max_ep"]):
        s = env.reset()
        ep_r = 0.
        for t in range(PARAMS["max_step"]):
            if not args.hide:
                env.render()
            a = rl.act(s)
            a = np.clip(np.random.normal(a, var), *env.action_bound)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)
            ep_r += r
            aloss, closs = 0, 0
            if rl.memory_full:
                var = max(var * 0.99992, min_var)
                aloss, closs = rl.learn()

            s = s_
            if done or t == PARAMS["max_step"]-1:
                if not rl.memory_full:
                    print("collecting data... Ep: %d/%d" % (ep, PARAMS["max_ep"]))
                else:
                    print('Ep: %d/%d | %s | ep_r: %.1f | var: %.2f | step: %i | aloss=%.4f | closs=%.4f' % (
                        ep, PARAMS["max_ep"], '---' if not done else 'done',
                        ep_r, var, t+1, aloss, closs))
                    break
        if rl.memory_full and ep % 20 == 0:
            rl.save_weights("{}/ep{}".format(MODEL_DIR, ep))


def eval():
    if args.load == -1:
        ep = PARAMS["max_ep"]-1
    else:
        ep = args.load
    rl.load_weights("{}/ep{}".format(MODEL_DIR, ep))
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.act(s)
        s, r, done = env.step(a)


if __name__ == "__main__":
    assert args.arms >= 2, ValueError("arms most >= 2")
    if args.arms == 2:
        PARAMS = {"training": args.load == 0, "n_arms": 2, "max_ep": 701, "max_step": 150,
                  "soft_replace": True, "random_target": True, "tau": 0.002, "gamma": 0.8, "lr": 0.0001,
                  "memory_capacity": 9000}
    else:
        PARAMS = {"training": args.load == 0, "n_arms": args.arms, "max_ep": 1001, "max_step": 150,
                  "soft_replace": True, "random_target": True, "tau": 0.001, "gamma": 0.8, "lr": 0.0001,
                  "memory_capacity": 9000}

    # set env
    print(PARAMS)
    env = ArmEnv(n_arms=PARAMS["n_arms"], random_goal=PARAMS["random_target"], on_mouse=False if PARAMS["training"] else True, show_fps=args.show_fps, fps=args.fps)
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    # set RL method (continuous)   
    rl = DDPG(a_dim, s_dim, a_bound, soft_replace=PARAMS["soft_replace"], tau=PARAMS["tau"], gamma=PARAMS["gamma"], lr=PARAMS["lr"], )

    MODEL_DIR = "models/{}arms".format(PARAMS["n_arms"])
    if PARAMS["training"]:
        train()
    else:
        eval()