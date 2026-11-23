from telepath_worker import spec

def test_specs():
    envsp = spec.EnvSpec()
    envsp.load('specs2/spec_env_1.yaml')
    print(envsp.spec)

    ags = spec.AgentSpec()
    ags.load('specs_test/spec_agent_1.yaml')
    print(ags.spec)

    repsp = spec.ReplaySpec()
    repsp.load('specs_test/spec_replay_1.yaml')
    print(repsp.spec)

    algsp = spec.AlgoSpec()
    algsp.load('specs_test/spec_agent_algo_1.yaml')
    print(algsp.spec)

