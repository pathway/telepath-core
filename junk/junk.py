def main_pred_timer(ea):
    print("--- main: load replay pickle ")
    full_replay.df = storage.read_df_pickle_from_location(sett['replay_loc'])


    print("--- main: try out some preds")

    #np_obs=np.expand_dims( np.array([-0.0041023744, -0.20529355, -0.017539551, 0.3]), axis=0 )
    obs=np.array([ [-0.0041023744, -0.20529355, -0.017539551, 0.3],
                   [+0.0041023744, 0.20529355, 0.017539551, 0.2],
                   [+0.0041023744, -0.20529355, 0.017539551, 0.1],] )


    eval_obs_subsample = full_replay.sample(1000, env=None).observations

    ensemble_q = ea._get_ensemble_output(eval_obs_subsample)
    print(ensemble_q)
    member_mean_deltas = (ensemble_q - ensemble_q.mean(axis=0))

    member_mean_deltas_neg = member_mean_deltas.clone()
    member_mean_deltas_neg[ member_mean_deltas_neg<0]=0
    member_mean_deltas_neg_std_score = member_mean_deltas_neg.std(axis=1).std(axis=1).cpu().numpy()

    member_mean_deltas_pos = member_mean_deltas.clone()
    member_mean_deltas_pos[ member_mean_deltas_pos>0]=0
    member_mean_deltas_pos_std_score = member_mean_deltas_pos.std(axis=1).std(axis=1).cpu().numpy()

    member_std_score = ( ensemble_q  - ensemble_q.mean(axis=0) ).std(axis=1).std(axis=1).cpu().numpy()

    member_obs_std_over_actions = ( ensemble_q  - ensemble_q.mean(axis=0) ).std(axis=2)
    member_action_std_over_obs  = ( ensemble_q  - ensemble_q.mean(axis=0) ).std(axis=1)

    member_std_score    # evaluate each member by how far its predictions are from mean
    # consider dropping members who are furthest from mean

    #obs=np.array([ [-0.0041023744, -0.20529355, -0.017539551, 0.3]] )

    import time

    res = ea.members[0].predict(obs, deterministic=False)

    try_pred_obs=0
    if try_pred_obs:
        for i in range(0,2):
            start = time.time()
            res = ea.predict( obs, deterministic=False )
            #print(res)
            end = time.time()
            print("det ms", (end - start)*1000.0)


        '''
        2ms for single pred or for 100 obs.
        Doesnt seem to matter which mode.
        '''

        for i in range(0,5):
            start = time.time()
            ensemble_mode = 'median'
            action, _states = ea.predict_mode(obs, deterministic=True, mode_selector='max', mode_ens=ensemble_mode)
            #print(res)
            end = time.time()
            print("det ms", (end - start)*1000.0)

        start = time.time()
        res = ea.predict( obs, deterministic=False )
        print(res)
        end = time.time()
        print("NOT det ms", (end - start)*1000.0)

    # CPU:
    # ensemble size  1: 0.3ms
    # ensemble size  3: 0.6ms
    # ensemble size  5: 1.0ms
    # ensemble size 10: 1.9ms
    # ensemble size 20: 3.5ms
    #
    # CUDA:
    # ensemble size  5: 1.7ms

def main_junk1():

    action, info = batch_agent.predict( obs )
    print("action",action,info)
    batch_agent.replay_buffer.df['q']=0


    # --- fill in q/v/adv/pi to dataframe for analysis

    q_list = []
    val_list = []
    adv_list = []
    pi_list = []
    for i in range(0,len(batch_agent.replay_buffer.df.index)):
        row = batch_agent.replay_buffer.df.iloc[i,:]
        print(batch_agent.replay_buffer.df.iloc[i,:])
        #print(row.obs)
        #action, info = batch_agent.predict( row.obs )
        #print("action",action,info)

        obs = row.obs.reshape((-1,) + sett['env'].observation_space.shape)
        obs = th.as_tensor(obs) #.to(self.device)
        with th.no_grad():
            qs, action = batch_agent.policy.q_net._predict_raw(obs,)

        #qs = batch_agent.policy._predict( th.Tensor(row.obs), deterministic=True )
        #print("qs action",qs,action)
        npa = np.array( qs.numpy()[0] )
        #batch_agent.replay_buffer.df.q[i] = npa
        q_list += [ npa, ]
        val_list += [ max(npa) ]
        adv_list += [ npa-max(npa) ]
        pi_list += [ action.numpy()[0] ]

    batch_agent.replay_buffer.df['q'] = q_list
    batch_agent.replay_buffer.df['v'] = val_list
    batch_agent.replay_buffer.df['a'] = adv_list
    batch_agent.replay_buffer.df['pi'] = pi_list
    print(batch_agent.replay_buffer.df[['q','v','a','pi']])

    sett['env'].close()

