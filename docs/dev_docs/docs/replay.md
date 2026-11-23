# replay buffers

## PdPlainBuffer
- uses pandas and csv
- allows insert

## VxBuffer
- supposedly better for large files
- easy calculated columns with built-in eval-type thing


## usage

#### specifying
- spec file


#### preparing
- write in fast format?
- 
- randomize?

#### storing samples
- see  common/off_policy_algorithm.py

```
replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_, done)
# format of these?
```

#### fetching minibatch samples for training
dqn/batch_dqn.py
```
replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

# format of result?
```


#### reading


#### writing

