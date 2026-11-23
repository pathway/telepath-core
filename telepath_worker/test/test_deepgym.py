from deepgym import deepgym_nets


def test_deepgym_nets_construct():
    n_actions = 2
    n_obs = 3
    device_string = "cuda:0"
    model = deepgym_nets.DeepGym_ModelNet(n_actions=n_actions, n_obs=n_obs, device_string=device_string)
    model = model.to(device_string)
    print(model)
    return model


if __name__=="__main__":
    test_deepgym_nets_construct()
