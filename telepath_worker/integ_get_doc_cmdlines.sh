cat commands.txt | sed 's/--/\\\n     --/g' | sed 's/\/tmp\/simplegym_seq_short_//g' | sed 's/specs2\/spec_replay_simplegym_seq_short.yaml/replay_spec.yaml/g' | sed 's/specs2\///g'
