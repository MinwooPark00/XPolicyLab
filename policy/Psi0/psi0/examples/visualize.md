# Data Visualization

If you want to visualize the first episode in a task to get a sense how the task look like. We provide a script for visualizing an episode.

## Install `Pinocchio` package

manually install pinocchio due to numpy version conflicts
```
uv pip install "pin>=3.8.0"
```

revert numpy version
```
uv pip install numpy==1.26.4
```

## Visualize 


### Visualize Real-World data
```
export task=Pick_bottle_and_turn_and_pour_into_cup

python scripts/viz/viz_episode_real.py \
  --args.data-dir=$PSI_HOME/data/real/$task \
  --args.port=9000 \
  --args.episode_idx=0
```

Open the link [http://localhost:9000/](http://localhost:9000/) in the broswer.

> The page displays g1 whole body joint state frame by frame and two red balls denote the control target wrist poses.

### Visualize SIMPLE data

```
export task=G1WholebodyLocomotionPickBetweenTablesVariant5-v0

python scripts/viz/viz_episode_simple.py \
  --args.data-dir=$PSI_HOME/data/simple/$task \
  --args.port=9000 \
  --args.episode_idx=0
```
