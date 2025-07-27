# RBE501-RL-arm-project
Motion planning of a robot arm using reinforcement learning

## Setup
### Base and Simulation Dependencies
First [install poetry](https://python-poetry.org/docs/)

Install default dependencies

``` shell
poetry install
```

### Optionally Install Hardware Support
Only do this if the hardware environment will be used to evaluate a trained
policy on hardware.

Clone the lerobot repository relative to this repository at `../lerobot`:

``` shell
cd ..
git clone https://github.com/huggingface/lerobot.git
```

Commit `c940676bdda5ab92e3f9446a72fafca5c550b505` of the lerobot repo has been
successfully tested on hardware so check out this commit:

``` shell
cd lerobot
git checkout c940676bdda5ab92e3f9446a72fafca5c550b505
```

Go back to the RL arm repository and install the hardware dependencies:

``` shell
cd ../RBE501-RL-arm-project/
poetry install --with hardware
```

## Example: Training a Policy
First start a poetry shell, if one currently does not exist:
``` shell
poetry shell
```

Train the policy using default argument values (this uses the PPO algorithm and
policy). Training time depends on hardware. 

``` shell
python train.py train
```
Visualize training reward progress using tensorboard. Create a separate poetry shell and run:

``` shell
tensorboard --logdir=logs/train/
```
Then open the address output to the console in a webbrowser.

## Example: Visualize a Trained Policy
First start a poetry shell, if one currently does not exist:

``` shell
poetry shell
```

Assuming a PPO policy has been trained using the example above, a visualization
of the policy controlling the robot in simulation can be generated. The model
number is output to the console at the start of training and also appears in the
log directory name, which can be seen in tensorboard:

``` shell
python train.py --vis eval --model-num <trained model number>
```
