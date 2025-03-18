# PFNs

Prior-data Fitted Networks (PFNs, https://arxiv.org/abs/2112.10510) are transformer encoders trained to perform supervised in-context learning on datasets randomly drawn from a prior.
Our priors can in general be described by a function that samples a datasets, or more generally a batch of datasets.
The PFN is then trained to predict a hold-out set of labels, given the rest of the dataset.

The pseudo code for a simple prior that would yield a PFN that does 1d ridge regression on datasets with 100 elements, could be something like this:

```python
def get_dataset_sample():
    x = RandomUniform(100,1)
    a = RandomNormal()
    b = RandomNormal()
    y = a * x + b
    return x, y
```

Check out our [tutorial](https://colab.research.google.com/drive/12YpI99LkuFeWcuYHt_idl142DqX7AaJf) to train your own ridge regression PFN.

### Install with pip

This way of installing allows you to use the package everywhere and still be able to edit files.
You should use a python version **>=3.10 and <=3.11**.
```bash
git clone https://github.com/automl/PFNs.git
cd PFNs
pip install -e .
```

### Get Started

Check out our [Getting Started Colab](https://colab.research.google.com/drive/12YpI99LkuFeWcuYHt_idl142DqX7AaJf).

### Tabular Data


For loading the pretrained TabPFN transformer model for classification and use it for evaluation, you can download the model like this

```python
import torch
from pfns.scripts.tabpfn_interface import TabPFNClassifier
# Load pretrained-model
classifier = TabPFNClassifier(base_path='.', model_string="prior_diff_real_checkpoint_n_0_epoch_42.cpkt")

train_xs = torch.rand(100,2)
test_xs = torch.rand(100,2)
train_ys = train_xs.mean(1) > .5
# Fit and evaluate
task_type = 'multiclass'
classifier.fit(train_xs, train_ys)
if task_type == 'multiclass':
    prediction_ = classifier.predict_proba(test_xs) # For survival [:, 1:]
else:
    prediction_ = classifier.predict(test_xs)
```


### BO

There is a BO version of this repo, with pretrained models at [github.com/automl/PFNs4BO](github.com/automl/PFNs4BO).
The two repos share a lot of the code, but the other is not anymore actively maintained.
You can also train your own models with our tutorial notebook [here](Tutorial_Training_for_BO.ipynb).

To run all BayesOpt experiments, please install this package with the `benchmarks` option:
```bash
pip install -e .[benchmarks]
```

### Bayes' Power for Explaining In-Context Learning Generalizations

This repository contains the code for the paper "Bayes' Power for Explaining In-Context Learning Generalizations".

Install in editable mode:
```bash
pip install -e .
```

We have a set of notebooks in this repository to reproduce the results of our paper.

- To reproduce the main ICL experiments, use the notebook `discrete_bayes.ipynb`.
- To run the Tiny-MLP generalization experiments, where we evaluate extrapolation, use the notebook `Tiny_MLP_Generalization.ipynb`.
- To run the Coin-Flipping experiments, where we show that the true posterior converges to the wrong probability, use the notebook `Cointhrowing_converging_to_wrong_posterior.ipynb`.
- To see the GP converging to the wrong solution for a step function, use the notebook `GP_fitting_a_step.ipynb`.


### Cite the work

PFNs were introduced in
```
@inproceedings{
    muller2022transformers,
    title={Transformers Can Do Bayesian Inference},
    author={Samuel M{\"u}ller and Noah Hollmann and Sebastian Pineda Arango and Josif Grabocka and Frank Hutter},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=KSugKcbNf9}
}
```

Training PFNs on tabular data (TabPFN) was enhanced in
```
@inproceedings{
  hollmann2023tabpfn,
  title={Tab{PFN}: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Noah Hollmann and Samuel M{\"u}ller and Katharina Eggensperger and Frank Hutter},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=cp5PvcI6w8_}
}
```

The BO version of PFNs was introduced in
```
@article{muller2023pfns,
  title={PFNs4BO: In-Context Learning for Bayesian Optimization},
  author={M{\"u}ller, Samuel and Feurer, Matthias and Hollmann, Noah and Hutter, Frank},
  journal={arXiv preprint arXiv:2305.17535},
  year={2023}
}
```

The "Bayes' Power for Explaining In-Context Learning Generalizations" is
```
@article{muller2024bayes,
  title={Bayes' Power for Explaining In-Context Learning Generalizations},
  author={M{\"u}ller, Samuel and Hollmann, Noah and Hutter, Frank},
  journal={arXiv preprint arXiv:2410.01565},
  year={2024}
}
```

