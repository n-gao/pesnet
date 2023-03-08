# Potential Energy Surface Network (PESNet)

Reference implementation of PESNet from <br>

<b>[Ab-Initio Potential Energy Surfaces by Pairing GNNs with Neural Wave Functions](https://openreview.net/forum?id=apv504XsysP)</b> <br/>
by Nicholas Gao, Stephan Günnemann<br/>
published as Spotlight at ICLR 2022.

and Planet and PESNet++ from

<b>[Sampling-free Inference for Ab-Initio Potential Energy Surface Networks](https://openreview.net/forum?id=Tuk3Pqaizx)</b> <br>
by Nicholas Gao, Stephan Günnemann <br>
published at ICLR 2023

## Run the code
First install [JAX](https://github.com/google/jax) and the correct [CUDA Toolkit](https://anaconda.org/anaconda/cudatoolkit) and [CUDNN](https://anaconda.org/anaconda/cudnn), then this package via
```bash
pip install -e .
```
You can now train a model, e.g., H2, via a config file
```bash
python train.py with configs/systems/h2.yaml print_progress=True
```
You can overwrite parameters either via [CLI](https://sacred.readthedocs.io/en/stable/command_line.html) or via the config file.
All progress is tracked on tensorboard.

## Reproduce the experiments
We encourage the use of `seml` to manage all experiments but we also supply commands to run the experiments directly.

### PESNet++ ablation study on N2
With `seml`:
```bash
seml n2_ablation add train_n2_ablation.yaml start
```
Without `seml`:
```bash
# PESNet
python train.py with configs/systems/n2.yaml \\
    init_method=pesnet \\
    pesnet.ferminet_params.activation=tanh \\
    pesnet.ferminet_params.input_config.mlp_activation=tanh \\
    pesnet.ferminet_params.jastrow_config=None \\
    pesnet.ferminet_params.determinants=32
# PESNet++ (default config)
python train.py with configs/systems/n2.yaml \\
    init_method=pesnet \\
    pesnet.ferminet_params.activation=silu \\
    pesnet.ferminet_params.input_config.mlp_activation=silu \\
    pesnet.ferminet_params.jastrow_config.n_layers=3 \\
    pesnet.ferminet_params.jastrow_config.activation=silu \\
    pesnet.ferminet_params.determinants=32
```

### Potential Energy Surfaces
To run all experiments from the PlaNet paper with `seml` simply run:
```bash
seml pes add train_pes.yaml start
```

## Contact
Please contact [gaoni@in.tum.de](mailto:gaoni@in.tum.de) if you have any questions.

## Cite
Please cite our paper if you use our method or code in your own works:
```
@inproceedings{gao_pesnet_2022,
    title = {Ab-Initio Potential Energy Surfaces by Pairing GNNs with Neural Wave Functions},
    author = {Gao, Nicholas and G{\"u}nnemann, Stephan}
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2022}
}
```
```
@inproceedings{gao_planet_2023,
    title = {Sampling-free Inference of Ab-initio Potential Energy Surface Networks},
    author = {Gao, Nicholas and G{\"u}nnemann, Stephan}
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2023}
}
```

## License
Hippocratic License v2.1
