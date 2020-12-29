import numpy as np
import pyglet


class Arm:
    def __init__(self, length=100., local_angle=np.pi/6, previous_arm=None):
        self.length = length
        self.thickness = 5
        self.local_angle = local_angle
        self.previous_arm = previous_arm
        self.next_arm = None
        if previous_arm is not None:
            previous_arm.add_next(self)

    @property
    def start_pos(self):
        if self.previous_arm is None:
            return np.zeros((2, ), dtype=np.float32)
        else:
            return self.previous_arm.end_pos

    @property
    def global_angle(self):
        if self.previous_arm is None:
            base_angle = 0
        else:
            base_angle = self.previous_arm.global_angle
        return (base_angle + self.local_angle) % (np.pi * 2)

    @property
    def end_pos(self):
        g_angle = self.global_angle
        return np.array([np.cos(g_angle), np.sin(g_angle)]) * self.length + self.start_pos

    def vertices(self):
        sp = self.start_pos
        ep = self.end_pos
        tr = np.pi / 2 - self.global_angle
        xy01 = sp + np.array([-np.cos(tr), np.sin(tr)]) * self.thickness
        xy02 = sp + np.array([np.cos(tr), -np.sin(tr)]) * self.thickness
        xy11 = ep + np.array([np.cos(tr), -np.sin(tr)]) * self.thickness
        xy12 = ep + np.array([-np.cos(tr), np.sin(tr)]) * self.thickness
        return np.concatenate((xy01, xy02, xy11, xy12))

    def add_next(self, arm):
        self.next_arm = arm

    def previous(self):
        return self.previous_arm


class ArmEnv(object):
    dt = .1    # refresh rate
    action_bound = [-2., 2.]

    def __init__(self, n_arms=2, random_goal=False, on_mouse=False, fps=False):
        self.viewer = None
        self.fps = fps
        self.viewer_width = 400
        self.viewer_height = 400
        assert n_arms > 0, ValueError
        self.arms = [Arm(200/n_arms, np.pi/6, None)]
        for i in range(n_arms-1):
            self.arms.append(Arm(200/n_arms, np.pi/6, self.arms[-1]))
        self.arm_pos_shift = np.array([self.viewer_width/2, self.viewer_height/2])  # arm1 start window center
        self.on_goal = 0
        self.goal_pos = np.array([125, 125], np.float32)
        self.goal_length = 40
        self.random_goal = random_goal
        self.on_mouse = on_mouse
        self.state_dim = self._state_reward()[0].shape[0]
        self.action_dim = n_arms

    def _state_reward(self):
        dxdy_ = np.empty((len(self.arms), 2), dtype=np.float32)
        for i in range(len(self.arms)):
            theta = self.arms[i].global_angle - np.pi / 2
            dxdy = self.goal_pos - (self.arms[i].end_pos + self.arm_pos_shift)
            dxdy_[i] = np.matmul(np.expand_dims(dxdy, axis=0), np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]))

        self_dist = np.sqrt(np.sum(np.square(dxdy_), axis=1))
        self_angle = np.arctan(dxdy_[:, 1] / (dxdy_[:, 0] + 1e-7))
        self_angle = np.where(dxdy_[:, 0] < 0, self_angle + np.pi, self_angle)
        self_angle = np.where((dxdy_[:, 0] > 0) & (dxdy_[:, 1] < 0), self_angle + 2*np.pi, self_angle)
        dist_norm = self_dist / np.sqrt(self.viewer_height ** 2 + self.viewer_width ** 2)
        angle_norm = (self_angle - np.pi)/np.pi
        s = np.concatenate(
            (dist_norm, angle_norm,
             [1. if self.on_goal else -1.]), axis=0)

        r = -dist_norm[-1]/5
        return s, r

    def step(self, action):
        if action.ndim > 1:
            action = np.squeeze(action, axis=0)
        done = False
        _action = np.clip(action, *self.action_bound)

        for i in range(len(self.arms)):
            arm = self.arms[i]
            arm.local_angle = (arm.local_angle + _action[i] * self.dt) % (np.pi * 2)

        s, r = self._state_reward()

        # done and reward
        goal_half_l = self.goal_length / 2
        shifted_last_pos = self.arms[-1].end_pos + self.arm_pos_shift
        if (self.goal_pos[0] - goal_half_l < shifted_last_pos[0] < self.goal_pos[0] + goal_half_l
        ) and (self.goal_pos[1] - goal_half_l < shifted_last_pos[1] < self.goal_pos[1] + goal_half_l):
            r += 1.
            self.on_goal += 1
            if self.on_goal > 30:
                done = True
        else:
            self.on_goal = 0

        return s, r, done

    def reset(self):
        if not self.on_mouse:
            if self.random_goal:
                r = np.pi * 2 * np.random.rand()
                d = self.viewer_width / 2 * np.random.rand()
                self.goal_pos[0] = np.cos(r) * d
                self.goal_pos[1] = np.sin(r) * d
                self.goal_pos += self.arm_pos_shift
            else:
                # random arms
                for arm in self.arms:
                    arm.local_angle = 2 * np.pi * np.random.rand()
        self.on_goal = 0
        return self._state_reward()[0]

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arms, self.goal_pos, self.goal_length, self.on_mouse,
                                 self.viewer_width, self.viewer_height, fps=self.fps)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(len(self.arms))-0.5    # two radians


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arms, goal_pos, goal_length, on_mouse=False, width=400, height=400, fps=False):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=width, height=height, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arms = arms
        self.goal_pos = goal_pos
        self.goal_length = goal_length
        self.on_mouse = on_mouse
        self.fps = fps
        self.center_coord = np.array([width/2, height/2])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', self._goal_vec()),
            ('c3B', (86, 109, 249) * 4))    # color
        self.barms = [self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,)  # color
        ) for _ in range(len(self.arms))]

        self.fps_display = pyglet.window.FPSDisplay(window=self)

    def _goal_vec(self):
        hl = self.goal_length / 2
        x, y = self.goal_pos[0], self.goal_pos[1]
        return [
            x - hl, y - hl,  # location
            x - hl, y + hl,
            x + hl, y + hl,
            x + hl, y - hl]

    def render(self, dt=None):
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        if self.fps:
            self.fps_display.draw()

    def on_mouse_motion(self, x, y, dx, dy):
        if self.on_mouse:
            self.goal_pos[0], self.goal_pos[1] = x, y

    def _update(self):
        shift = np.concatenate([self.center_coord for _ in range(4)])
        for arm, barm in zip(self.arms, self.barms):
            barm.vertices = arm.vertices() + shift
        self.goal.vertices = self._goal_vec()


if __name__ == '__main__':
    env = ArmEnv(n_arms=2, random_goal=False)
    env.reset()
    while True:
        env.render()
        env.step(env.sample_action())