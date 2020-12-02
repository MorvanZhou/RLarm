"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG


def train():
    # start training
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            if rl.memory_full:
                env.render()
            a = rl.act(s)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)
            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %d/%d | %s | ep_r: %.1f | step: %i | learning: %s' % (
                    ep, MAX_EPISODES, '---' if not done else 'done', ep_r, j, "yes" if rl.memory_full else "no"))
                break
        if ep % 50 == 0:
            rl.save_weights("models/ep{:03d}".format(ep))


def eval():
    rl.load_weights("models")
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if __name__ == "__main__":
    MAX_EPISODES = 1001
    MAX_EP_STEPS = 200
    ON_TRAIN = True
    SOFT_REPLACE = True
    RANDOM_TARGET = True
    ON_MOUSE = False

    # set env
    env = ArmEnv(random_target=RANDOM_TARGET, on_mouse=ON_MOUSE)
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    # set RL method (continuous)
    rl = DDPG(a_dim, s_dim, a_bound, soft_replace=SOFT_REPLACE)

    if ON_TRAIN:
        train()
    else:
        eval()