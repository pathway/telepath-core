# Tensorboard

During training, metrics describing the training process are written to s3, which can be monitored by [tensorboard, the visualization app provided by tensorflow](https://www.tensorflow.org/tensorboard).  

(Yes, telepath is based on pytorch not tensorflow for the deep learning layer, but tensorboard can serve either).

Tensorboad log location is specified by __tblog_folder__ in the [project spec](specs.md).

To use tensorboard:

```
./telepath tb  --project_spec=telepath_settings.yaml   
```

Then connect to container port 6006.


## Local tensorboard

If you can access the log folder in s3 from your local box, you can also [install tensorboard locally](https://pypi.org/project/tensorboard/):
```
pip install tensorboard
```

Then run:

```
tensorboard --logdir=<tblog_folder>
```
Then access tensorboard via `http://localhost:6006/`
