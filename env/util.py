import os

class Recorder:
    def __init__(self):
        pass

    def begin_episode(self, path):
        raise NotImplementedError

    def record(self, data):
        raise NotImplementedError

    def end_episode(self):
        raise NotImplementedError

class GameboardRecorder(Recorder):
    def __init__(self, agent_num):
        super().__init__()
        self.agent_num = agent_num
        self.task_fp = None
        self.agent_fp = None

    # helpful to drawing
    def begin_episode(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.task_fp = open(path + '/task.log', 'w')
        self.agent_fp = []
        for i in range(self.agent_num):
            name = path + '/agent{:03d}.log'.format(i)
            self.agent_fp.append(open(name, 'w'))

    def record(self, data):   
        for d in data['task']:
            self.task_fp.write(','.join(map(str, d)) + '\n')
        
        for agfp, d in zip(self.agent_fp, data['agent']):
            agfp.write(','.join(map(str, d)) + '\n')

    def end_episode(self):
        self.task_fp.close()
        for f in self.agent_fp:
            f.close()
