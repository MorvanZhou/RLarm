import pyglet
import time


class BaseViewer(pyglet.window.Window):
    def __init__(self, width, height, caption, on_mouse=False, show_fps=False, fps=-1):
        super().__init__(width=width, height=height, resizable=False, caption=caption, vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

        self.batch = pyglet.graphics.Batch()
        self.labels = {}

        self.on_mouse = on_mouse
        self.on_mouse = on_mouse
        self.fps = fps
        self.show_fps = show_fps

        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.keyboard = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keyboard)

    def render(self, *args, **kwargs):
        t0 = time.time()
        self.update(*args, **kwargs)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
        if self.fps > 0:
            duration = time.time()-t0
            default_duration = 1/self.fps
            if default_duration > duration:
                time.sleep(default_duration - duration)

    def update(self, *args, **kwargs):
        raise NotImplemented

    def on_draw(self):
        self.clear()
        self.batch.draw()
        [v.draw() for v in self.labels.values()]
        if self.show_fps:
            self.fps_display.draw()