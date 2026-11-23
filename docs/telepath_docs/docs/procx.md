# Multiple processes with procx

## Parallelization strategies

You may wish to train multiple policies at once. This can be useful for ensembling, checking for stability, or exploring hyperparameters.

If using the same hyperparameters / algo_spec and replay, then multiple Telepath training processes can easily be run on a single container.

Another strategy is to run multiple independent containers, this allows for different hyperparameters.

Both of these can also be combined: multiple containers, each with multiple training processes.

Generally, the _train_ command is the only one that benefits from parallelization.

## procx

Before using procx you should confirm:
  
- you have sufficient RAM and GPU RAM to support multiple processes.  You need an estimate of single-process usage first.
- you have sufficient CPUs for the process count
- your job output files will not clobber each other.  For computing agents, see [Unique Agent Names](uniq.md) below.

## command line

Just prepend the telepath train command with the procx as shown:

```
procx [process count] [process command + arguments]
```

Example:
```
procx 4 telepath train --project_spec=...
```

