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
You should use a python version **>=3.9**.
```bash
git clone https://github.com/automl/PFNs.git
cd PFNs
pip install -e .
```

### Get Started

Check out our [Getting Started Colab](https://colab.research.google.com/drive/12YpI99LkuFeWcuYHt_idl142DqX7AaJf).

For loading the pretrained TabPFN transformer model for classification and use it for evaluation:
```python

# Load pretrained-model
current_path = ..
classifier = PFNClassifier(base_path=current_path, model_string="prior_diff_real_checkpoint_n_0_epoch_42.cpkt")

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