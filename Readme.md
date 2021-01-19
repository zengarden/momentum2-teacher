# Momentum<sup>^2</sup> Teacher: Momentum Teacher with Momentum Statistics for Self-Supervised Learning

# Requirements
1. All experiments are done with python3.6, torch==1.5.0; torchvision==0.6.0

# Usage

## Training
To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:

1. using `-d` to specify gpu_id for training, e.g., `-d 0-7`
2. using `-b` to specify batch_size, e.g., `-b 256`
3. using `--experiment-name` to specify the output folder, and the training log & models will be dumped to './outputs/${experiment-name}'
4. using `-f` to specify the description file of ur experiment.

e.g., 
```
python3 momentum_teacher/tools/train.py -b 256 -d 0-7 --experiment-name your_exp -f momentum_teacher/exps/arxiv/exp_8_v100/momentum2_teacher_100e_exp.py
```


## Linear Evaluation:

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8 gpus machine, run:

1. using `-d` to specify gpu_id for training, e.g., `-d 0-7`
2. using `-b` to specify batch_size, e.g., `-b 256`
3. using `--experiment-name` to specify the folder for saving pre-training models.
```
python3 momentum_teacher/tools/eval.py -b 256 --experiment-name your_exp -f momentum_teacher/exps/arxiv/linear_eval_exp_byol.py
```


# Results

## Results of Pretraining on a Single Machine


After pretraining on 8 NVIDIA V100 GPUS and 1024 batch-sizes, the results of linear-evaluation are:

|pre-train code                                                                   |pre-train</br> epochs| pre-train time | accuracy     | weights|
|---------------------------------------------------------------------------      |---------------------|  ------------- |  --------    | ------ |
| [path](momentum_teacher/exps/arxiv/exp_8_v100/momentum2_teacher_100e_exp.py)    | 100                 |  ~1.8 day      | 70.7         |   -    |
| [path](momentum_teacher/exps/arxiv/exp_8_v100/momentum2_teacher_200e_exp.py)    | 200                 |  ~3.6 day      | 72.7         |   -    |
| [path](momentum_teacher/exps/arxiv/exp_8_v100/momentum2_teacher_300e_exp.py)    | 300                 |  ~5.5 day      | 73.8         |   -    |

After pretraining on 8 NVIDIA 2080 GPUS and 256 batch-sizes, the results of linear-evaluation are:

|         pre-train code                                                                        |pre-train</br> epochs| pre-train time | accuracy         | wights |
|-----------------------------------------------------------------------------------------------|---------------------|  ------------- |  --------        | ------ |
|[path](momentum_teacher/exps/arxiv/exp_8_2080ti/momentum2_teacher_100e_32batch_1mm_099_exp.py) | 100                 |  ~2.5 day      | 70.4             |  -     |
|[path](momentum_teacher/exps/arxiv/exp_8_2080ti/momentum2_teacher_200e_32batch_1mm_099_exp.py) | 200                 |  _             | coming soon      |  -     |
|[path](momentum_teacher/exps/arxiv/exp_8_2080ti/momentum2_teacher_300e_32batch_1mm_099_exp.py) | 300                 |  _             | coming soon      |  -     |

## Results of Pretraining on Multiple Machines

E.g., To do unsupervised pre-training with 4096 batch-sizes and 32 V100 GPUs. run:


Suggesting that each machine has 8 V100 GPUs and there are 4 machines
```
# machine 1:
export MACHINE=0; export MACHINE_TOTAL=4; python3 momentum_teacher/tools/train.py -b 4096 -f xxx
# machine 2:
export MACHINE=1; export MACHINE_TOTAL=4; python3 momentum_teacher/tools/train.py -b 4096 -f xxx
# machine 3:
export MACHINE=2; export MACHINE_TOTAL=4; python3 momentum_teacher/tools/train.py -b 4096 -f xxx
# machine 4:
export MACHINE=3; export MACHINE_TOTAL=4; python3 momentum_teacher/tools/train.py -b 4096 -f xxx
```

results of linear-eval:

|         pre-train code                                                                        |pre-train</br> epochs| pre-train time  | accuracy     | weights|
|-----------------------------------------------------------------------------------------------|---------------------|  -------------  |  --------    | ------ |
|[path](momentum_teacher/exps/arxiv/exp_128_2080ti/momentum2_teacher_100e_4096batch_16mm_exp.py)| 100                 |  ~11hour        |  70.3        |   -    |
|[path](momentum_teacher/exps/arxiv/exp_128_2080ti/momentum2_teacher_200e_4096batch_16mm_exp.py)| 200                 |  ~22hour        | coming soon  |   -    |
|[path](momentum_teacher/exps/arxiv/exp_128_2080ti/momentum2_teacher_300e_4096batch_16mm_exp.py)| 300                 |  ~33hour        | coming soon  |   -    |



To do unsupervised pre-training with 4096 batch-sizes and 128 2080 GPUs, pls follow the above guides. Results of linear-eval:

|         pre-train code                                                                        |pre-train</br> epochs| pre-train time | accuracy     | weights|
|-----------------------------------------------------------------------------------------------|---------------------|  ------------- |  --------    | ------ |
|[path](momentum_teacher/exps/arxiv/exp_128_2080ti/momentum2_teacher_100e_4096batch_16mm_exp.py)| 100                 |  ~5hour        | 69.0         |   -    |
|[path](momentum_teacher/exps/arxiv/exp_128_2080ti/momentum2_teacher_200e_4096batch_16mm_exp.py)| 200                 |  ~10hour       | 71.5         |   -    |
|[path](momentum_teacher/exps/arxiv/exp_128_2080ti/momentum2_teacher_300e_4096batch_16mm_exp.py)| 300                 |  ~15hour       | 72.1         |   -    |


# Data Preparation
Prepare the ImageNet data in `${root_of_your_clone}/data/imagenet_train`, `${root_of_your_clone}/data/imagenet_val`.
Since we have an internal platform(storage) to read imagenet, I have not tried the local mode. 
You may need to do some modification in `momentum_teacher/data/dataset.py` to support the local mode.


