import numpy as np
import torch as th
import telepath_utils
import spec




class EnsembleAgent(object):
    '''
    Agent implements ensemble-mean and Thompson sampling methods
    '''
    members = []
    env = None

    member_selector = None
    mode_selector = 'max'
    mode_ens = 'mean'

    agentid = None

    def __init__(self, env):
        self.env = env
        self.agentid = telepath_utils.get_rand_agentid()
        pass


    def _get_ensemble_output(self, obs ):
        ens_action = []
        ens_q_list = []
        ens_state = []
        for idx, m in enumerate(self.members):
            obs = obs.reshape((-1,) + self.env.observation_space.shape)
            obs = th.as_tensor(obs) #.to(self.device)
            with th.no_grad():
                action, q_values, state = m.predict_raw(obs)
                ens_action.append(action)
                ens_q_list.append(q_values)
                ens_state.append(state)
        ens_q = th.stack(ens_q_list)
        return ens_q

    def get_ens_selector(self):
        ens_selector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        return ens_selector

    def predict(self, obs, deterministic=True):
        """
        deterministic=True: select action with "best" Q
        mode then determins how the "best" Q is determined.

        deterministic=False: thompson sampling

        Same interface as stable-baselines agent at least for predict
        What about loading?
        """
        action_count = self.env.action_space.n
        obs_count = obs.shape[0]

        ens_q = self._get_ensemble_output(obs)

        if self.member_selector:
            # cut down to selected subset
            ens_selector_th = th.tensor(self.member_selector)
            ens_q[ens_selector_th == 1]

        if deterministic:
            if self.mode_ens=='median':
                ret = ens_q.median(axis=0)[0]
            elif self.mode_ens=='mean':
                ret = ens_q.mean(axis=0)[0]
            elif self.mode_ens=='max':
                ret = ens_q.max(axis=0)[0]
            elif self.mode_ens=='min':
                ret = ens_q.min(axis=0)[0]
            elif self.mode_ens=='std':
                ret = ens_q.std(axis=0)
            else:
                ret = ens_q.mean(axis=0)[0]

            if self.mode_selector =='max':
                ret2 = ret.argmax() # default to mean
            elif self.mode_selector =='min':
                ret2 = ret.argmin() # default to mean

            return ret2.cpu().reshape(-1).numpy(), None
        else:
            #if self.mode_selector == 'upper_thompson':
            #    max_a = np.random.randint(0, action_count, size=obs_count)
            #    return max_a, None    # do not assume single observation

            if self.mode_selector=='uniform':
                max_a = np.random.randint(0, action_count, size=obs_count)
                return max_a, None    # do not assume single observation

            else:
                # if mode=='thompson':
                #action_member_selector = th.tensor(np.random.choice(ens_q.shape[0], size=ens_q.shape[2]) )

                member_selector = th.tensor(np.random.choice(ens_q.shape[0], 1) )
                ens_q = ens_q[member_selector]

                ret = ens_q.mean(axis=0)[0]  # dont really need a mean because its only one selected...

                if self.mode_selector =='max':
                    ret2 = ret.argmax() # default to mean
                elif self.mode_selector =='min':
                    ret2 = ret.argmin() # default to mean

                return ret2.cpu().reshape(-1).numpy(), None




class AgentWrapper(object):

    agent = None        # agent object whatever that is
    agentid = None


    def __init__(self, env):
        self.env = env
        self.agentid = telepath_utils.get_rand_agentid()
        pass


    def predict(self, obs, deterministic=True):
        return self.agent.predict( obs, deterministic )


    def read(self, agent_spec_filename):
        aspec = spec.AgentSpec()
        aspec.load(agent_spec_filename)
        if spec['agent_type'] == 'ensemble':
            # construct the agent
            # get member list
            # load the members
            pass
        elif spec['type']=='sbl3_batchdqn':
            # load normal way
            pass


    def write(self, agent_spec_filename):
        # write spec
        # depending on type, write files
        pass
