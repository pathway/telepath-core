from telepath_worker import spec
from telepath_worker import ext_replay
from pprint import pprint as pp

replay_csv_file_small='test/test_data/generated_tiny_cartpole.csv'
replay_csv_file_small_spec='specs2/spec_replay_cartpole_tiny.yaml'

replay_csv_file_client='test/test_data/client_example_01.csv'
replay_csv_file_client_spec='specs2/spec_replay_client_01.yaml'

replay_csv_file_huge='test/test_data/generated.csv'  # see generate_replay.py


def __test_specreplay_explicit():
    # dont need this anymore
    repsp = spec.ReplaySpec()
    repsp.load('specs2/spec_replay_cartpole_tiny.yaml')
    print(repsp.spec)

    r = ext_replay.VxBuffer(replay_csv_file_small)
    r.sample_strategy='seq'
    r.fieldmap = repsp.spec['fieldmap']
    r.read_from_location()

    r.traj_order()
    print(r.df)

    # must happen after read
    if 'col_formula_reward' in repsp.spec:
        reward_col_name="virt_reward"
        r.df.add_virtual_column(reward_col_name, repsp.spec['col_formula_reward'] )
        r.fieldmap['reward'] = reward_col_name

    print(r.sample(1))
    return r


def test_specreplay_implicit_pd():
    r = ext_replay.PdPlainBuffer(replay_csv_file_small)
    r.sample_strategy='seq'
    r.set_spec(replay_csv_file_small_spec)
    pp(r.specs.spec)
    # can now safely read and sample; virtual cols added here
    #r.read_from_location()

    print(r.df)
    print(r.sample(1))

    print( r.get_col_subsets() )
    pp( r.get_col_validity() )

    gym = r.get_specgym()
    print(gym)
    print(gym.action_space)
    print(gym.observation_space)

    return r


def test_specreplay_implicit_vx():
    r = ext_replay.VxBuffer(replay_csv_file_small)
    r.sample_strategy='seq'
    r.set_spec(replay_csv_file_small_spec)
    pp(r.specs.spec)
    # can now safely read and sample; virtual cols added here
    r.read_from_location()

    print(r.df)
    print(r.sample(1))

    print( r.get_col_subsets() )
    pp( r.get_col_validity() )

    gym = r.get_specgym()
    print(gym)
    print(gym.action_space)
    print(gym.observation_space)

    return r


def test_specreplay_client_vx():
    r = ext_replay.VxBuffer(replay_csv_file_client)
    r.sample_strategy='seq'
    r.set_spec(replay_csv_file_client_spec)

    pp(r.specs.spec)
    # can now safely read and sample; virtual cols added here
    r.read_from_location()
    r.traj_order()
    #print(r.df[ r.df['received_email'].isin([1]) ] )

    #print(r.sample(1))

    print( r.get_col_subsets() )
    pp( r.get_col_validity() )

    gym = r.get_specgym()
    print(gym)
    print(gym.action_space)
    print(gym.observation_space)

    #r.print_colstats()
    return r




if __name__=="__main__":
    test_specreplay_implicit_pd()
