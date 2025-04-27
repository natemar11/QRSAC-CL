# QRSAC
Implementation of Quantile Regression Soft Actor Critic (QRSAC) from "Outracing champion Gran Turismo drivers with deep reinforcement learning" by Peter R. Wurman, Samuel Barrett, Kenta Kawamoto, James MacGlashan, Kaushik Subramanian, Thomas J. Walsh, Roberto Capobianco, Alisa Devlic, Franziska Eckert, Florian Fuchs, Leilani Gilpin, Piyush Khandelwal, Varun Kompella, HaoChih Lin, Patrick MacAlpine, Declan Oller, Takuma Seno, Craig Sherstan, Michael D. Thomure, Houmehr Aghabozorgi, Leon Barrett, Rory Douglas, Dion Whitehead, Peter Dürr, Peter Stone, Michael Spranger & Hiroaki Kitano. [[Paper]](https://www.nature.com/articles/s41586-021-04357-7). 

This repository is based on [RLkit](https://github.com/vitchyr/rlkit) and [DSAC](https://github.com/xtma/dsac), two popular reinforcement learning frameworks implemented in PyTorch.

The github code link: [[Code]](https://github.com/shilpa2301/QRSAC)

## Requirements
- python 3.10+
- All dependencies are available in requirements.txt and environment.yml

##Updates - how to run (for now)

First must download the donkeycarsimulator
Get the linux version of the binaries from the link on this repository [Repo](https://github.com/tawnkramer/gym-donkeycar)

cd into folder with simulator/DonkeySimLinux in one terminal and run
'''
./donkey_sim.x86_64
'''
Alternatively, to run without the GUI (will run slightly faster), use the command
'''
./donkey_sim.x86_64 -batchmode -nographics -port 9091
'''

## Usage (run after starting donkey simulator)
You can write your experiment settings in configs/your_config.yaml and run with 
```
python qrsac.py --config your_config.yaml --gpu 0 --seed 0
```
Set `--gpu -1`, your program will run on CPU.

## Experiments
2 different experiments are conducted to validate the working of the QRSAC algorithm - on a donkercar simulator and on a real-world scaled RC car (Jetracer).

### Experiments on DonkeyCar

<img src='./readme_media/DonkeyCar.gif'>

### Experiments on JetRacer

<img src='./readme_media/Jetracer.gif'>



