from hyperdash import Experiment

from agent.trainer import Trainer

from agent.util import EpsilonExponentialConst

from env.my_env import Gameboard

from env.util import GameboardRecorder

from agent.my_dqn import SimpleDQN


name = 'icaart_v0'
exp = Experiment(name)

agent_num = 6
task_num = 100
view = 4
env = Gameboard(agent_num, task_num, view)
env.seed(1106)

# also modified target_update,10--->100
params = {
    'name'              : "name",
    'episodes'          : 5000,
    'steps'             : 1000,
    'no_op_episodes'    : 100,
    'epsilon'           : EpsilonExponentialConst(init=1.0, rate=0.9992, thr=0.1),
    'train_every'       : 4,
    'save_model_every'  : 5000,
    'is_centralized'    : False,

    'agent_num'         : agent_num,
    'env'               : env,
    'action_space'      : env.action_space,
    'observation_space' : env.observation_space,
    'preprocess'        : None,
    'recorder'          : GameboardRecorder(agent_num),

    'agent': [
        SimpleDQN(
            action_space      = env.action_space,
            observation_space = env.observation_space,
            observation_space_local = env.observation_space_local,
            memory_size       = 10000,
            batch_size        = 64,
            learning_rate     = 0.0001,
            gamma             = 0.95,
            target_update     = 500 * 10,
            use_dueling       = False
        ) for _ in range(agent_num)
    ],

    'hyperdash': exp
}


if __name__ == "__main__":
    # print("aaaa")
    trainer = Trainer(**params)
    trainer.train()


    exp.end()
