# Environment Management

We try to maintain a central code repository for $\Psi_0$ as well as all the baselines.

Key considerations are:

+ We want to train each baseline with minimum code change, so we create separate isolated environment for each baseline. 
+ At the same time, we want all the baseline models, dataloaders, etc can be mixed in the same environment.


So we manage the environments as illuated below:

<p align="center">
  <img src="../assets/media/envs.png" alt="envs" />
</p>

To conclude:

1. $\Psi_0$ is managed using the default `pyproject.toml`, and depdendency groups include `base`, `psi`, `viz` and `serve`.
2. Each baseline install `src/` package and base dependencies, so all the baseline models, dataloaders etc., can be used in the envrionment.
3. Each baseline maintain a seperate `requirements.txt` which include the baseline's `torch` and `transformers` versions.

