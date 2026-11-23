# Multi-engine

- Goal: allow the use of different RL engines
- Method: create facade / factory pattern

## Engines
- stables-baselines3 OK
- fqf TODO
- BCQ TODO
- hybrid TODO


## Agent Interface

ISSUES:
- batch/online dichotomy
- different algos have different settings
- 

```

class FrameworkAgent_DiscretaAction_Offpolicy:

    # managed
    _agent = None

    # inputs / chosen
    batch_online = "BATCH"
    #path = None
    
    project_spec = None
    algo_spec = None  # agent_spec = None # ??
    #algo_spec_path = None

    # project_spec
    # tb_log_folder = None




class FrameworkAgent_DiscretaAction_Offpolicy_Sbl3:

    # classmethod; BATCH or ONLINE or BOTH
    def _get_agent( self, batch_online="BATCH", algo_, set_, spec_vec_env ):
        self.batch_online = batch_online
        if self.batch_online ==  "BATCH":
            self.agent_class=BatchDQN
            self._agent = _get_batch_agent( algo_, set_, spec_vec_env )
            return agent
        else:
            self.agent_class=DQN
            self._agent =  _get_online_agent( algo_, set_, spec_vec_env )
            return agent

    # how do we tell it which variant?
        # should spec know about this

    # classmethod
    def load(self, agent_path):
        self.agent = storage.load_agent_location(self.agent_class, agent_path )
        return self.agent 

    def save(self, agent_path):
        # calls agent.save
        storage.save_agent_location( self.agent, agent_path )

    def predict( self, obs, deterministic=True, qvals=False ):
        # _qvals, _info 
        action, _states, = self.agent.predict(obs, deterministic=deterministic)
        return { 'action': action, 'states': _states, }
        #  'qvals': _qvals, 'info': _info, 'confidence': _confidence

    def learn_online( self, params_dict ):
        # must be abstracted out
        return agent.learn( **params_dict )

    def learn_batch( self, total_timesteps, ):
                        tb_log_name='/tmp/tblog/',
                        #callback=eval_callback
                        )
        return self.agent.learn_batch( total_timesteps=trainsteps_add, ):
                        tb_log_name= tb_log_name,
                        #callback=eval_callback
                        )

    # get Q values ...


```


## References
```
# construct
def _get_batch_agent( algo_, set_, spec_vec_env ):
def _get_online_agent( algo_, set_, vec_env ):

# predict
action, _states = agent.predict(obs, deterministic=deterministic)

# loading
batch_agent = storage.load_agent_location(BatchDQN, agent_path)
agent = agent_class.load(path)

# saving
storage.save_agent_location( batch_agent, agent_path )
agent.save(path)

# learn
online_agent.learn( log_interval=10,  # extra
                            total_timesteps=algo_['total_samples'],
                            eval_env=vec_env_sim,  # extra
                            #callback=eval_callback,
                            tb_log_name='s_dqn_'+online_agent.agentid  )

# learn batch
batch_agent.learn_batch(total_timesteps=trainsteps_add,
                        tb_log_name= tb_log_name,
                        #callback=eval_callback
                        )

```

