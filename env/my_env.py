import numpy as np

class Gameboard(object):
    '''
    add limitation: at most carry 1 ball
    actions: 
    0: up; 1:right; 2:down; 3:left (task-executin mode)
    when agents are at the waiting zone, one of the actions turns into "wait"

    observation has two parts:
    local for seeing tasks
    global for seeing Goal and self-position,other agents

    queue data structure to decide task execution order 
    '''
    def __init__(self,agent_num,task_num,view=4):
        '''
        agent_num: int
        task_num : int
        view     : int
        '''
        self.height = 20
        self.width  = 20
        self.view   = view
        self.agent_num = agent_num
        self.task_num  = task_num
        self.LIMIT  =  1

        # shape (space)
        self.action_space = 4
        self.state_space  = (self.height,self.width)
        self.observation_space = (3, self.height, self.width)#global observation(3,20,20)
        self.observation_space_local = (1, 2*self.view+1, 2*self.view+1)#(1,9,9)

        #variables
        self.step_cnt = 0
        self.agent_pos = np.zeros((self.agent_num,2),dtype=np.int16)
        self.agent_state = np.zeros(self.state_space, dtype=np.int16)#(20,20)
        self.task_state = np.zeros(self.state_space, dtype=np.int16)#(20,20)
        self.action = np.array([-1]*self.agent_num,dtype=np.int16)
        self.reward = np.zeros((self.agent_num,),dtype=np.float32)
        self.task_carry = np.zeros((self.agent_num,), dtype=np.int16)
        self.move_map = [[-1,0],[0,1],[1,0],[0,-1]]
        self.task_update = []
        # newly added variables 11-2
        # wait zone's index also correspond to wait action
        self.wz = [[self.height//2+1,self.width//2],
                   [self.height//2,self.width//2-1],
                   [self.height//2-1,self.width//2],
                   [self.height//2,self.width//2+1]]#waiting zone

        self.reset()

    def reset(self):
        self.task_remain = self.task_num
        self.step_cnt = 0
        self.action = np.array([-1]*self.agent_num,dtype=np.int16)
        self.reward = np.zeros((self.agent_num,),dtype=np.float32)
        self.task_carry = np.zeros((self.agent_num,), dtype=np.int16)

        self.agent_pos = np.zeros((self.agent_num,2),dtype=np.int16)
        self.agent_state = np.zeros(self.state_space, dtype=np.int16)
        self.task_state  = np.zeros(self.state_space)
        self.task_update = []
        #11-2 new added
        self.que = []#task execution queue

        arr = np.arange(self.height * self.width)
        arr = np.delete(arr,self.height//2 * self.width + self.width//2)
        sample = np.random.choice(arr, self.agent_num + self.task_num, replace=False)
        for i in range(self.agent_num + self.task_num):
            y = sample[i] // self.width
            x = sample[i] %  self.width
            if i < self.agent_num:
                self.agent_pos[i] = [y,x]
                self.agent_state[y,x] = 1
            else:
                self.task_state[y,x] = 1
                self.task_update.append([self.step_cnt,-1,y,x])
        return self._get_observation()

    def step(self, action):
        assert len(action) == self.agent_num
        assert action.dtype == 'int16'

        self.step_cnt += 1

        self.action = action
        self.reward = np.zeros((self.agent_num,), dtype=np.float64)
        done = False

        task_finished = []#this var is from yamazaki
        action_order = np.arange(self.agent_num)
        np.random.shuffle(action_order)
        for i in action_order:
            if (list(self.agent_pos[i]) in self.wz) \
                and (action[i]==self.wz.index(list(self.agent_pos[i]))) \
                and (self.task_carry[i]>0):
                self.reward[i] = -0.005
                continue
            # update agent position and agent_state
            npos = self.agent_pos[i] + self.move_map[action[i]]
            if not npos[0]<0 and not npos[0]>=self.height \
                and not npos[1]<0  and not npos[1]>=self.width \
                and not self.agent_state[npos[0],npos[1]]:
                self.agent_state[self.agent_pos[i][0], self.agent_pos[i][1]] = 0
                self.agent_pos[i] = npos
                self.agent_state[npos[0],npos[1]] = 1
            else:
                self.agent_state[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
                self.reward[i] = -0.001

            y,x = self.agent_pos[i]
            if (list(self.agent_pos[i]) in self.wz) and (self.task_carry[i]>0):
                #if sum(self.agent_state[z[0],z[1]] for z in self.wz if z!=self.agent_pos[i])==0 :
                self.que.append(i)
            if self.task_state[y, x]:
                if self.task_carry[i] < self.LIMIT:
                    self.reward[i] = 0.05
                    self.task_state[y,x] = 0
                    self.task_carry[i] += 1
                    task_finished.append([i,y,x])
            if self.task_remain < 0.01:
                print('you fool ! task_num is:' , self.task_remain)
                done = True
        for i,y,x in task_finished:
            self.task_update.append([self.step_cnt,i,y,x])
        if self.que:
            i = self.que.pop(0)
            self.reward[i] = 1.0 * self.task_carry[i]
            self.task_remain -= self.task_carry[i]
            self.task_carry[i] = 0
            print('score +++1 !!')
        
        cfcnt = len(self.que)
        return np.array(self._get_observation()), np.array(self.reward, dtype=np.float64), done, cfcnt

    def render(self, mode='log'):
        ret = None
        if mode == 'log':
            ret = {
                'agent':[[
                    self.step_cnt,
                    i,
                    self.agent_pos[i][0],
                    self.agent_pos[i][1],
                    self.action[i],
                    self.reward[i]
                    ] for i in range(self.agent_num)],
                    'task': list(self.task_update)
            }
            self.task_update.clear()
        elif mode == 'state':
            ret = np.zeros((self.height,self.width),dtype=np.int16)
            ret += self.agent_state
            ret += self.task_state * 2
        return ret
        

    def _get_observation(self):
        '''0:self-pos; 1:task; 2:other agent pos; 3:hole;
           return group obs
        '''
        gobs = []
        #obs = {}
        # itvg = self.observation_space[0]#global view interval, int
        # itvl = self.observation_space_local[0]#int
        for agent_id in range(self.agent_num):
            obs_global = np.zeros(self.observation_space) # single obs global
            obs_local  = np.zeros(self.observation_space_local) #single obs local
            obs = {}
            if self.task_carry[agent_id]:
                obs_global[0,self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 1 #self pos
                obs_global[1, self.height//2, self.width//2] = 1 #hole
                obs_global[2] = self.agent_state # other agents
                obs_global[2, self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 0
            else:
                obs_local[0] = self._get_local(self.task_state, self.agent_pos[agent_id]) # ?? ok?
                obs_global[0, self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 1
                obs_global[2] = self.agent_state
                obs_global[2, self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 0
            obs['global'] = obs_global
            obs['local']  = obs_local

            gobs.append(obs) # a list of dict
        # obs_global = np.zeros((self.agent_num * itvg, self.height, self.width))#(6*3,20,20)
        # obs_local = np.zeros((self.agent_num * itvl, 2*self.view+1, 2*self.view+1))#(6*1,9,9)
    
        # for agent_id in range(self.agent_num):
        #     obs = {}#single obs
        #     if self.task_carry[agent_id]:
        #         #ignore task
        #         obs_global[agent_id * itvg, self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 1
        #         obs_global[agent_id * itvg + 1, self.height//2, self.width//2] = 1
        #         obs_global[agent_id * itvg + 2] = self.agent_state
        #         obs_global[agent_id * itvg + 2, self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 0
        #     else:
        #         # ignore Goal
        #         obs_local[agent_id * itvl] = self._get_local(self.task_state, self.agent_pos[agent_id])
        #         obs_global[agent_id * itvg, self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 1
        #         obs_global[agent_id * itvg + 2] = self.agent_state
        #         obs_global[agent_id * itvg + 2, self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 0
        # obs_local = obs_local.reshape((self.agent_num,)+self.observation_space_local)
        # obs_global= obs_global.reshape((self.agent_num,)+self.observation_space)
        # obs['local'] = obs_local
        # obs['global']= obs_global
        #a = [np.reshape(obs[:4], self.observation_space), np.reshape(obs[4:], self.observation_space)]
        return gobs

    def _get_local(self, state, pos):
        y = pos[0]
        x = pos[1]
        local = np.zeros((2*self.view+1,2*self.view+1))
        low_y = max([0,y-self.view])
        high_y= min([self.height,y+self.view+1])
        low_x = max([0,x-self.view])
        high_x= min([self.height,x+self.view+1])
        local[:high_y-low_y,:high_x-low_x] = state[low_y:high_y, low_x:high_x].copy()
        local = local.reshape(self.observation_space_local)
        return local


    def seed(self, seed=0):
        np.random.seed(seed=seed)
