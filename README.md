# PFNs


### Install with pip

This way of installing allows you to use the package everywhere and still be able to edit files.
```bash
pip install . -e
```

### Get Started

Check out our [Getting Started Notebook](Getting_Started_With_PFNs.ipynb).


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
