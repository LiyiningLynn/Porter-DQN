import os
import sys
import shutil
from datetime import datetime as dt
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# only for logging 'episode,reward,loss'
class Logger:
    def __init__(self, path):
        self.log_fp = open(path, 'w')
        self.log_fp.write('episode,reward,loss,conflict,finish\n')

    def log(self, episode, reward, loss, cfcnt, finish):
        self.log_fp.write('{:d},{:.6f},{:.6f},{:d},{:d}\n'.format(episode, reward, loss, cfcnt,finish))

class Trainer:
    def __init__(self, name, env, agent, episodes, steps,
                 no_op_episodes, epsilon, train_every, save_model_every,
                 agent_num, action_space, observation_space,
                 recorder=None, record_every=4, preprocess=None,
                 is_centralized=False, obs_num=1, hyperdash=None):

        # Base saving path
        tstr = dt.now().strftime('%Y%m%d_%H%M%S')
        self.base_path = 'outputs/' + name + '_{}'.format(tstr)
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # Copy script to record the parameters
        script_path = sys.argv[0]
        shutil.copy(script_path, self.base_path + '/script.py')

        self.train_record_path = self.base_path + '/train'
        if not os.path.exists(self.train_record_path):
            os.makedirs(self.train_record_path)

        self.eval_record_path = self.base_path + '/eval'
        if not os.path.exists(self.eval_record_path):
            os.makedirs(self.eval_record_path)

        self.model_path = self.base_path + '/model'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.train_logger = Logger(self.base_path + '/train.csv')
        self.eval_logger  = Logger(self.base_path + '/eval.log')

        self.env               = env
        self.agent             = agent #a group of 6 agents
        self.name              = name
        self.episodes          = episodes
        self.steps             = steps
        self.no_op_episodes    = no_op_episodes
        self.epsilon           = epsilon
        self.train_every       = train_every
        self.save_model_every  = save_model_every
        self.agent_num         = agent_num
        self.action_space      = action_space
        self.observation_space = observation_space
        self.recorder          = recorder
        self.record_every      = record_every
        self.preprocess        = preprocess
        self.is_centralized    = is_centralized
        self.obs_num           = obs_num
        self.hyperdash         = hyperdash

    def _episode(self, epsilon, do_train=False, do_memorize=False, record_path=""):
        obs_queue = np.zeros((self.agent_num, self.obs_num,) + self.env.observation_space)
        # the beginning of an episode, reset the environment
        obs = self.env.reset() # should be group obs
        print('self.env.task_num: ',self.env.task_num)

        if self.obs_num > 1:
            obs_queue[:, 0] = obs
            obs = np.array(obs_queue)

        if self.preprocess is not None:
            obs = self.preprocess(obs)

        # I don't record task.log and agent.log TEMPORILY
        # Now I do it
        # 2019-11-21 only record conflict, same as previous exp
        # 2019-11-28 log again
        if record_path != "":
            self.recorder.begin_episode(record_path)
            self.recorder.record(self.env.render())

        total_reward = 0.0
        total_loss = 0.0
        train_cnt = 0
        ave_loss=0
        cfcnt = 0
        for s in range(self.steps):
            if self.is_centralized:
                action = self.agent.get_action(obs, epsilon)
            else:
                # obs[i] should be a dict
                action = [ag.get_action(obs[i], epsilon) for i, ag in enumerate(self.agent)]
            nobs, reward, done, cf = self.env.step(np.array(action,dtype='int16'))
            cfcnt += cf
            total_reward += reward.sum()
            
            if done:
                break
                
            if self.obs_num > 1:
                obs_queue = np.roll(obs_queue, 1, axis=1)
                obs_queue[:, 0] = nobs
                nobs = np.array(obs_queue)
            for i, ag in enumerate(self.agent):
                if do_memorize:
                    ag.memory.add(obs[i], action[i], reward[i], nobs[i])
                if do_train and (s + 1) % self.train_every == 0:
                    total_loss += ag.train()
                    train_cnt += 1

            obs = nobs
            if record_path != "":
                self.recorder.record(self.env.render())
        if record_path != "":
            self.recorder.end_episode()
            
        ave_loss = total_loss / train_cnt if train_cnt > 0 else 0
        finish = self.env.get_finished()
        print("eps: {}".format(str(epsilon)))
        print('total reward: ', total_reward)
        return total_reward, ave_loss, cfcnt, finish

    def train(self):
        # Run agents with random actions to collect experiences
        print('Gathering random experiences...', end = '', flush=True)
        for e in range(self.no_op_episodes):
            self._episode(epsilon=1.0, do_train=False, do_memorize=True)
        print(' done!', flush=True)

        # Main loop
        for e in range(self.episodes):
            print('***** episode {:d} *****'.format(e + 1))

            ### Train ###
            print('--- train ---')
            epsilon = self.epsilon.get(e)
            print('epsilon: {:.4f}'.format(epsilon))

            path = self.train_record_path + '/episode{:05d}'.format(e + 1) \
                if (e + 1) % self.record_every == 0 else ""
            total_reward, ave_loss, cfcnt, finish = self._episode(epsilon=epsilon,
                do_train=True, do_memorize=True, record_path=path)

            print('total reward: {:.2f}, average loss: {:.4f}, conflict: {:d}, finish: {:d}'.format(total_reward, ave_loss, cfcnt, finish))
            self.train_logger.log(e, total_reward, ave_loss, cfcnt, finish)
            if self.hyperdash is not None:
                self.hyperdash.metric("train_reward", total_reward, log=False)
            
            # Save model
            if (e + 1) % self.save_model_every == 0:
                if self.is_centralized:
                    self.agent.save(self.model_path, e + 1)
                else:
                    for i, ag in enumerate(self.agent):
                        ag.save(self.model_path, e + 1, i)

            print()

        print('Saved data at {:s}'.format(self.base_path))
