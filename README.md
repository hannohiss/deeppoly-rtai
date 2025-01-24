# RTAI 2024 Course Project

Certifying NNs, joint work with [@h-buechi](https://github.com/h-buechi), [@mashaprostotak](https://github.com/mashaprostotak).

## Setup instructions

We recommend you install a [Python virtual environment](https://docs.python.org/3/library/venv.html) to ensure dependencies are the same as the ones we will use for evaluation.
To evaluate your solution, we are going to use Python 3.10.
You can create a virtual environment and install the dependencies using the following commands:

```bash
$ virtualenv venv --python=python3.10
$ source venv/bin/activate
$ pip install -r requirements.txt
```

If you prefer conda environments we also provide a conda `environment.yaml` file which you can install (After installing [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html)) via

```bash
$ conda env create -f ./environment.yaml
$ conda activate rtai-project
```

for `mamba` simply replace `conda` with `mamba`.

If you prefer the nice package manager `pixi` simply do the following:

```bash
$ pixi install
$ pixi run tests  # run all tests
```

## Running the verifier

We will run your verifier from `code` directory using the command:

```bash
$ python code/verifier.py --net {net} --spec test_cases/{net}/img_{dataset}_{eps}.txt
```

In this command, 
- `net` is equal to one of the following values (each representing one of the networks we want to verify): `fc_linear, fc_base, fc_w, fc_d, fc_dw, fc6_base, fc6_w, fc6_d, fc6_dw, conv_linear, conv_base, conv6_base, conv_d, skip, skip_large, skip6, skip6_large`.
- `dataset` is the dataset name, i.e.,  either `mnist` or `cifar10`.
- `eps` is the perturbation that the verifier should certify in this test case.

To test your verifier, you can run, for example:

```bash
$ python code/verifier.py --net fc_base --spec test_cases/fc_base/img_mnist_0.048839.txt
```

To evaluate the verifier on all networks and sample test cases, we provide an evaluation script.
You can run this script from the root directory using the following commands:

```bash
chmod +x code/evaluate.sh
./code/evaluate.sh
```
