# DRL-CEWS
Additional materials for paper "Curiosity-Driven Energy-Efficient Worker Scheduling in Vehicular Crowdsourcing: A Deep Reinforcement Learning Approach" accepted by ICDE 2020.

## :page_facing_up: Description
DRL-CEWS is a novel deep reinforcement learning (DRL) approach for curiosity-driven energy-efficient worker scheduling, to achieve an optimal trade-off between maximizing the collected amount of data and coverage fairness, and minimizing the overall energy consumption of workers.

### Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/DRL-CEWS.git
    cd DRL-CEWS
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```

## :zap: Quick Inference

Test the model trained with 100 PoIs (2 UAVs and 2 charging stations). 
Download the model from [Google Driver](https://drive.google.com/drive/folders/1GUA9FzH5oR8egUE4BNifA3OOi7UtAngw?usp=sharing) to `ckpt/`. Then, change the `trainable` to `False` in the parameter configuration file `/uav2_charge2/exper_dppo_curiosity/params.py`. After, run the following command the test the model. 

```
python run.py
```
Last, find the result under `/result`.


## :computer: Training
Change the `trainable` to `True` in the parameter configuration file `/uav2_charge2/exper_dppo_curiosity/params.py` and then run the following command the train the model. 
```
python run.py
```
Find the result under `/result`.

## :checkered_flag: Testing
Same as that in Quick Inference.

## :scroll: Acknowledgement
This paper was supported by National Natural Science
Foundation of China (No. 61772072).

## :e-mail: Contact

If you have any question, please email `ynzhao@bit.edu.cn`.

## Paper
If you are interested in our work, please cite our paper as

```
@inproceedings{liu2020curiosity,
  title={Curiosity-driven energy-efficient worker scheduling in vehicular crowdsourcing: A deep reinforcement learning approach},
  author={Liu, Chi Harold and Zhao, Yinuo and Dai, Zipeng and Yuan, Ye and Wang, Guoren and Wu, Dapeng and Leung, Kin K},
  booktitle={2020 IEEE 36th International Conference on Data Engineering (ICDE)},
  pages={25--36},
  year={2020},
  organization={IEEE}
}
```
