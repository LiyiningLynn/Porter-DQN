import time
import numpy as np
import tkinter as tk
from tkinter import Button
from PIL import ImageTk, Image
from num2words import num2words

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 50  # pixels
HEIGHT = 20  # grid height
WIDTH = 20  # grid width


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT + 2*UNIT, HEIGHT * UNIT + 2*UNIT))
        self.wait_img  = PhotoImage(Image.open("img/wait.png").resize((40, 40)))
        # self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT + 2*UNIT,
                           width=WIDTH * 2*UNIT)
        button_height = HEIGHT * UNIT + 75
        for h in range(HEIGHT+1):
            x1, y1, x2, y2 = UNIT, (h + 1) * UNIT, UNIT + WIDTH * UNIT, (h + 1) * UNIT
            canvas.create_line(x1, y1, x2, y2)

        for w in range(WIDTH+1):
            x1, y1, x2, y2 = (w + 1) * UNIT, UNIT, (w + 1) * UNIT, UNIT + HEIGHT * UNIT
            canvas.create_line(x1, y1, x2, y2)
        # Button
        play_button = Button(self, text="play", command=self.deleteimage)
        play_button.config(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH/5 * UNIT, button_height, window=play_button)

        pause_button= Button(self, text="pause", command=self.some_func)
        pause_button.config(width=10, activebackground="#33B5E5")
        canvas.create_window(2 * WIDTH/5 * UNIT, button_height, window=pause_button)

        #canvas.create_image(500,500,image=self.wait_img,tags='nino')

        tp = (12,7)
        #canvas.create_image(500,500,image=self.wait_img,tags='xx'+str(tp))
        num = 123
        canvas.create_image(500,500,image=self.wait_img,tags=num2words(num).replace(' ', ''))

        canvas.pack()

        return canvas
    
    def some_func(self):
        print('boring')

    def render(self):
        time.sleep(0.03)
        self.update()

    def deleteimage(self):
        tp = (12,7)
        num = 123
        #dele = self.canvas.delete(num2words(num).replace(' ', ''))
        dele = self.canvas.delete('kkk')
        print('what delete returns:',dele)
        #return dele


if __name__ == "__main__":
    env = Env()
    while True:
        env.render()