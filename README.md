# RTAI 2024 Course Project

Certifying NNs, joint work with [@h-buechi](https://github.com/h-buechi), [@mashaprostotak](https://github.com/mashaprostotak).

## Folder structure
In the directory `code`, you can find 4 files and one folder: 
- The file `networks.py` contains the fully-connected and convolutional neural network architectures used in this project.
    - The file `MLP.py` generates a sequential fully connected neural network and is used in `networks.py`.
    - The file `skip_block.py` generates a skip block (i.e. a residual skip connection) and is used in `networks.py`.
- The file `verifier.py` contains a template of the verifier. Loading of the stored networks and test cases is already implemented in the `main` function. If you decide to modify the `main` function, please ensure that the parsing of the test cases works correctly. Your task is to modify the `analyze` function by building upon DeepPoly convex relaxation. Note that the provided verifier template is guaranteed to achieve **0** points (by always outputting `not verified`).
- The file `evaluate` will run all networks and specs currently in the repository. It follows the calling convention we will use for grading.
- The folder `utils` contains helper methods for loading and initialization (There is most likely no need to change anything here).


In the directory `models`, you can find 17 neural networks (16 MNIST and 1 CIFAR) weights. These networks are loaded using PyTorch in `verifier.py`. Note that we included two `_linear` networks which do not contain activation functions.

In the directory `test_cases`, you can find 17 subfolders and the ground truth file `gt.txt`. Each subfolder is associated with one of the networks using the same name. In a subfolder corresponding to a network, you can find 2 test cases for each network. As explained in the lecture, these test cases **are not** part of the set of test cases that we will use for the final evaluation.

Note that all inputs are images with pixel values between 0 and 1. The same range also applies to all abstract bounds that we want to verify. 

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

If you prefer a nice package manager (pixi) simply do the following:

```bash
$ pixi install
$ pixi run tests
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