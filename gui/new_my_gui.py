import os
import time
import numpy as np
import tkinter as tk
from tkinter import Button
from PIL import ImageTk, Image
import pandas as pd
from num2words import num2words

running = True  # Global flag
idx = 0  # loop index

path = os.getcwd() + '/data/episode04968/'
agent_files = os.listdir(path)
agent_files.remove('task.log')
print(agent_files)
# 0: plain   1:carry  2:wait
# read file to dataframe
colnames1 = ['step','agID','h','w','action','rwd']
colnames2 = ['step','agID','h','w']
dfag = {}
for af in agent_files:
    df = pd.read_csv(path+af, header=None, names=colnames1)
    tag = num2words(int(df.iloc[0]['agID']))
    # print('tag is :',tag)
    dfag[tag] = df
dfood = pd.read_csv(path+'task.log', header=None, names=colnames2)

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 50  # pixels
HEIGHT = 20  # grid height
WIDTH = 20  # grid width

AGENT_NUM = len(dfag)
TASK_NUM  = len(dfood.loc[dfood['step'] == 0])
STEP      = dfag['zero'][dfag['zero']['step']>0].step# a Series from 1 to 1000, dtype int

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT + 2*UNIT, HEIGHT * UNIT + 2*UNIT))
        self.status = {}# All agents' status
        self.agents = {}# Canvas image
        # Create images
        self.agent_img = PhotoImage(Image.open("img/agent.png").resize((40, 40)))
        self.carry_img = PhotoImage(Image.open("img/carry.png").resize((40, 40)))
        self.wait_img  = PhotoImage(Image.open("img/wait.png").resize((40, 40)))
        self.food_img  = PhotoImage(Image.open("img/food.png").resize((40, 40)))

        self.canvas = self._build_canvas()

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT + 2*UNIT,
                           width=WIDTH * 2*UNIT)
        # Grid                  
        button_height = HEIGHT * UNIT + 75
        for h in range(HEIGHT+1):
            x1, y1, x2, y2 = UNIT, (h + 1) * UNIT, UNIT + WIDTH * UNIT, (h + 1) * UNIT
            canvas.create_line(x1, y1, x2, y2)
        for w in range(WIDTH+1):
            x1, y1, x2, y2 = (w + 1) * UNIT, UNIT, (w + 1) * UNIT, UNIT + HEIGHT * UNIT
            canvas.create_line(x1, y1, x2, y2)
        # Button
        play_button = Button(self, text="play", command=self.start)
        play_button.config(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH/5 * UNIT, button_height, window=play_button)

        pause_button= Button(self, text="pause", command=self.stop)
        pause_button.config(width=10, activebackground="#33B5E5")
        canvas.create_window(2 * WIDTH/5 * UNIT, button_height, window=pause_button)
        # Hole and waiting zone , index compatible with logic code
        canvas.create_rectangle((HEIGHT/2+1) * UNIT , (HEIGHT/2+1) * UNIT, 
                                (HEIGHT/2 + 2) * UNIT, (HEIGHT/2 + 2) * UNIT,
                                fill = '#320A70')
        canvas.create_rectangle((HEIGHT/2) * UNIT , (HEIGHT/2+1) * UNIT, 
                                (HEIGHT/2 + 1) * UNIT, (HEIGHT/2 + 2) * UNIT,
                                fill = '#F7E66B')
        canvas.create_rectangle((HEIGHT/2+2) * UNIT , (HEIGHT/2+1) * UNIT, 
                                (HEIGHT/2 + 3) * UNIT, (HEIGHT/2 + 2) * UNIT,
                                fill = '#F7E66B')
        canvas.create_rectangle((HEIGHT/2+1) * UNIT , (HEIGHT/2) * UNIT, 
                                (HEIGHT/2 + 2) * UNIT, (HEIGHT/2 + 1) * UNIT,
                                fill = '#F7E66B')
        canvas.create_rectangle((HEIGHT/2+1) * UNIT , (HEIGHT/2+2) * UNIT, 
                                (HEIGHT/2 + 2) * UNIT, (HEIGHT/2 + 3) * UNIT,
                                fill = '#F7E66B')
        # Add agents to canvas step0
        # Initially use agent image (plain)
        for tag,df in dfag.items():
            #tag of agent is agent ID
            print('in build canvas tag',tag)
            y = df.iloc[0]['h']#index at 0
            x = df.iloc[0]['w']
            self.agents[tag] = canvas.create_image((x+1.5)*UNIT, (y+1.5)*UNIT, image=self.agent_img, tags=tag)
            canvas.itemconfig(self.agents[tag], tags=tag)
            self.status[tag] = 0
        # Place tasks
        for _, row in dfood[dfood['step']==0].iterrows():
            #tag = 'task' + str(index)# tag of task == index of init state and distinguish from agent tag
            y = row['h']
            x = row['w']
            #tp = (x,y)
            #tag = 'task'+str(tp)
            canvas.create_image((x+1.5)*UNIT, (y+1.5)*UNIT, image=self.food_img, tags=self.get_tag(x,y))
            #print('build canvas tuple tags: ',tag)

        canvas.pack()
        return canvas
    
    def some_func(self):
        print('boring')
    
    def step(self, stp):
        self.canvas.delete(self.itext)# delete step text from previous frame
        self.render()
        self.show_step(stp)
        # move agents
        for i in range(AGENT_NUM):
            tag = num2words(i)
            print('in step tag', tag)
            df = dfag[tag]
            x = df.loc[df['step']==stp, 'w'].values[0]
            y = df.loc[df['step']==stp, 'h'].values[0]
            x0, y0 = self.canvas.coords(self.agents[tag])
            self.canvas.move(self.agents[tag], (x+1.5)*UNIT-x0, (y+1.5)*UNIT-y0)
            #self.canvas.tag_raise(self.agents[tag])#这行来自原始代码，不懂，先写上
            # 去他妈的，就用reward来作为依据好了
            if df.loc[df['step']==stp, 'rwd'].values[0]>0 and self.status[tag]==0:
                # change status from plain to carry
                self.canvas.delete(tag)
                self.agents[tag] = self.canvas.create_image((x+1.5)*UNIT, (y+1.5)*UNIT, image=self.carry_img, tags=tag)
                self.status[tag] = 1
                print(self.status)
                print('change!!')
            elif df.loc[df['step']==stp, 'rwd'].values[0]>0 and self.status[tag]!=0:
                #从carry或者wait到直接投球没有等待
                self.canvas.delete(tag)
                self.agents[tag] = self.canvas.create_image((x+1.5)*UNIT, (y+1.5)*UNIT, image=self.agent_img, tags=tag)
                self.status[tag] = 0
            elif df.loc[df['step']==stp, 'rwd'].values[0]==-0.005 and self.status[tag]==1:
                # from carry to wait
                self.canvas.delete(tag)
                self.agents[tag] = self.canvas.create_image((x+1.5)*UNIT, (y+1.5)*UNIT, image=self.wait_img, tags=tag)
                self.status[tag] = 2
            
            # if task here is picked
            con = dfood.loc[(dfood['w']==x)&(dfood['h']==y)]
            if (not con.empty) and (self.status[tag]==0):
                print("a task to be removed: ",y,x)
                #tp = (x,y)
                self.canvas.delete(self.get_tag(x,y))
                #print('step tuple: ',str(tp))
        print('status: ',self.status)
    def get_tag(self,x,y):
        tag = num2words(x).replace(' ', '')+num2words(y).replace(' ', '')
        return tag
               
    def render(self):
        time.sleep(0.3)
        #self.show_step()
        self.update()
        time.sleep(1)# 1秒是个很长的时间，这行先测试

    def show_step(self, step=0):
        itext_name =self.canvas.create_text(0.5 * UNIT ,1.5 * UNIT, text='step', font=("Purisa", 24))
        self.canvas.itemconfig(itext_name)
        self.itext=self.canvas.create_text(0.5 * UNIT ,2.5 * UNIT, text=str(step), font=("Purisa", 20))
        self.canvas.itemconfig(self.itext,text=str(step))

    def start(self):
        """Enable scanning by setting the global flag to True."""
        global running
        running = True

    def stop(self):
        """Stop scanning by setting the global flag to False."""
        global running
        running = False


if __name__ == "__main__":
    env = Env()
    env.show_step()
    while True:
        env.render()#位置应该没关系吧
        # iterate step from 1 to 1000
        for s in STEP:
            if running:
                env.render()
                env.step(s)
            
        break

# if __name__ == "__main__":
#     env = Env()
#     i = 0
#     while i < STEP:
#         if i == 0:
#             env.start()
#         env.render(i)
#         i += 1