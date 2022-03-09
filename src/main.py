import pyglet
from pyglet.gl import *
from pyglet.window import keys

pos = [0, 0, -20]
rot_y = 0


class Model:

    def get_text(self, file):
        tex = pyglet.image.load(file).texture
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        return pyglet.graphics.TextureGroup(tex)

    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        self.top = self.get_tex("img.png")

        tex_coords = ('t2f', (0, 0, 1, 0, 1, 1, 0, 1) * 4)

        x, y, z = 0, 0, 0
        X, Y, Z = 1, 1, 1
        self.batch.add(4, GL_QUADS, None,
                       ("v3f", (x, y, z, X, y, z, X, Y, z, x, Y, z)))

    def draw(self):
        self.batch.draw()


class Player:
    def __init__(self):
        self.pos = [0] * 3
        self.rot = [0] * 2

    def update(self, dt, keys):
        pass
class Window(pyglet.window.Window):

    def Projection(self):
        glMatrixMode(GL_PERSPECTIVE)
        glLoadIdentity()

    def Model(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def set3d(self):
        self.Projection()
        gluPerspective(70, self.width / self.height, 0.05, 1000)
        self.Model()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_minimum_size(200, 200)

        self.model = Model()
        self.player = Player()

    def on_draw(self):
        self.clear()
        self.set3d()
        self.model.draw()


if __name__ == "__main__":
    window = Window(width=500,
                    height=300,
                    caption="Truss Evolution",
                    resizable=True)
    glClearColor(0.5, 0.7, 1, 1)
    pyglet.app.run()