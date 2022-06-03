______________________________________________________________________
<div align="center">

# Polyphonic Music Modeling

<p align="center">
  <a href="https://github.com/AnetaJas">👋 Aneta Jaśkiewicz</a>
  <a href="https://github.com/arybs">👋 Arkadiusz Rybski</a>
</p>

______________________________________________________________________

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/wiktorlazarski/ai-awesome-project-template/blob/master/LICENSE)

</div>

## Installation with `pip`

Installation:

```bash
pip install git+https://github.com/AnetaJas/polyphonic-music-modeling.git
```

## Setup

```bash
# Clone repo
git clone https://github.com/AnetaJas/polyphonic-music-modeling.git

# Go to repo directory
cd polyphonic_music_modeling

# (Optional) Create virtual environment
python -m venv venv
source ./venv/bin/activate

# Install project in editable mode
pip install -e .[dev]

# (Optional but recommended) Install pre-commit hooks to preserve code format consistency
pre-commit install
```

## Setup with Anaconda or Miniconda

```bash
# Clone repo
git clone https://github.com/AnetaJas/polyphonic-music-modeling.git

# Go to repo directory
cd polyphonic_music_modeling

# Create and activate conda environment
conda env create -f ./conda_env.yml
conda activate polyphonic_music_modeling_env

# (Optional but recommended) Install pre-commit hooks to preserve code format consistency
pre-commit install
```


</div>

## Running scripts

In order to run scripts you have to activate your virutal envrionment. 

``` bash
source venv/bin/activate
```
To run experiment you have to configure config file (configs/train_experiment.yaml)
There are few possible params to configure:
* *dataset_module: dataset_root* is absolute path to directory with your dataset module. 
* *nn_module: lr* is learning rate (double)
* *training: max_epochs* - maximum number of epochs (int)
* *training: wand_project* - name of project in wandbai
* *training: model_name* - name of neural network to choose from {LSTM, GRU, R-Transformer(not working properly)}
* *training: model: layers* - number of layers of LSTM. GRU, R-Transformer (int)
* *training: model: embedding_dim* - int (GRU/LSTM)
* *training: model: hidden_dim* - int (GRU/LSTM)
* *training: model: d_model* - int (R-Transformer)
* *training: model: h* - int (R-Transformer)
* *training: model: ksize* - int (R-Transformer)
* *training: model: n* - int (R-Transformer)
* *training: model: n_level* - int (R-Transformer)
* *training: model: dropout* - double (R-Transformer)
* *test: path* - absolute path to model checkpoint file

After setting up config file you can run training like:
```bash
python scripts/training/train.py
```
If you have saved checkpoint file, and you only wanto run tests:
```bash
python scripts/training/test.py
```
## R-Transfomer
R-Transfomer model is taken from: [R-Trasnformer](https://github.com/DSE-MSU/R-transformer)
### References

@article{wang2019rtransf,
  title={R-Transformer: Recurrent Neural Network Enhanced Transformer},
  author={Wang, Zhiwei and Ma, Yao and Liu, Zitao and Tang, Jiliang},
  journal={arXiv preprint arXiv:1907.05572},
  year={2019}
}

## GRU 
GRU model consists of embedding layer, [gru layers](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) and linear layer.

## LSTM
LSTM model consists of embedding layer, [lstm layers](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) and linear layer.
