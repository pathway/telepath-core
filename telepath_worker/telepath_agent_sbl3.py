from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN, BatchDQN
from telepath_worker import telepath_utils
from telepath_worker.telepath_utils import dbg
import telepath_worker.storage_layer as storage
from telepath_common import *


def _get_batch_agent( algo_, set_, spec_vec_env ):
    '''
    algo_   algo_spec_content
    set_    setspec_content
    '''
    batch_agent = BatchDQN('MlpPolicy',
       spec_vec_env,
       verbose=2,
       learning_rate=algo_['agent_learning_rate'],
       gamma=algo_['env_gamma'],
       batch_size=algo_['agent_batch_size'],
       buffer_size=10,
       target_update_interval=algo_['agent_target_update_interval'],
       learning_starts=algo_['agent_learning_starts'],
       device=PYTORCH_DEVICE,
       tensorboard_log=set_['tblog_folder'] )
    batch_agent.agentid = telepath_utils.get_rand_agentid()
    return batch_agent


def _get_online_agent( algo_, set_, vec_env ):
    '''
    algo_   algo_spec_content
    set_    setspec_content
    '''
    online_agent = DQN('MlpPolicy',
        vec_env,
        verbose=0,
        learning_rate=algo_['agent_learning_rate'],
        gamma=algo_['env_gamma'],
        batch_size=algo_['agent_batch_size'],
        buffer_size=1000000,
        target_update_interval=algo_['agent_target_update_interval'],
        learning_starts=algo_['agent_learning_starts'],
        exploration_fraction=0.5,
        device=PYTORCH_DEVICE,
        tensorboard_log=set_['tblog_folder'],
        )
    online_agent.agentid = telepath_utils.get_rand_agentid()

    return online_agent


class TelepathAgent_DiscretaAction_Offpolicy:
    # managed
    _agent = None
    _agent_class = BatchDQN
    _agent_id = None

    # inputs / chosen
    batch_online = "BATCH"
    # path = None

    project_spec = None
    algo_spec = None  # agent_spec = None # ??
    # algo_spec_path = None

    # project_spec
    # tb_log_folder = None


    def getid(self):
        return self._agent_id



class TelepathAgent_DiscretaAction_Offpolicy_Sbl3( TelepathAgent_DiscretaAction_Offpolicy ):

    # classmethod; BATCH or ONLINE or BOTH
    def _init_agent(self, algo_, set_, spec_vec_env, batch_online="BATCH",):
        self.batch_online = batch_online
        if self.batch_online == "BATCH":
            self._agent_class = BatchDQN
            self._agent = _get_batch_agent(algo_, set_, spec_vec_env)
        else:
            self._agent_class = DQN
            self._agent = _get_online_agent(algo_, set_, spec_vec_env)

        self._agent_id = self._agent.agentid
        return self._agent
    # how do we tell it which variant?  should spec know about this


    # --- load/save

    # classmethod
    def load_agent(self, agent_path, _agent_class=BatchDQN):
        dbg("telepath_agent", comment="load_agent", data= agent_path )
        if _agent_class:
            self._agent_class=_agent_class
        self._agent = storage.load_agent_location(self._agent_class, agent_path)
        self._agent_id = self._agent.agentid
        return self._agent

    def save_agent(self, agent_path):
        dbg("telepath_agent", comment="save_agent", data= agent_path )
        # calls agent.save
        storage.save_agent_location(self._agent, agent_path)


    # --- buffer related

    def get_buffer(self):
        return self._agent.replay_buffer

    def get_buffer_pos(self):
        return self._agent.replay_buffer.pos

    def replace_buffer(self, newbuffer):
        del self._agent.replay_buffer
        self._agent.replay_buffer = newbuffer


    # --- learning

    def learn_online(self, params_dict):
        # must be abstracted out
        return self._agent.learn(**params_dict)

    def learn_batch(self, total_timesteps, tb_log_name = '/tmp/tblog/',):
        # callback=eval_callback )
        return self._agent.learn_batch(total_timesteps=total_timesteps,
            tb_log_name = tb_log_name,
            # callback=eval_callback
            )


    # --- prediction

    def predict(self, obs, deterministic=True, qvals=False):
        # _qvals, _info
        action, _states, = self._agent.predict(obs, deterministic=deterministic)
        #return {'action': action, 'states': _states, }
        return action, _states
        #  'qvals': _qvals, 'info': _info, 'confidence': _confidence

    def predict_raw(self, obs,   ):
        action, q_values, _states, = self._agent.predict_raw(obs,)
        return action, q_values, _states
