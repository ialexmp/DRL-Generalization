# Exploring Generalization in Deep Reinforcement Learning for Control Tasks

## Abstract

Deep reinforcement learning (DRL) has achieved significant success in addressing
various tasks; however, trained agents often struggle to generalize beyond
the environments in which they were initially trained. This thesis investigates the generalization capabilities of DRL algorithms through a series of independent experiments on control tasks of varying complexity. These tasks range from simple scenarios like Mountain Car to more complex challenges such as Pick and Place, utilizing the Gymnasium and Gymnasium-Robotics suites with the MuJoCo physics engine. Each experiment involves training a model using Proximal
Policy Optimization (PPO) within a specific environment and subsequently evaluating
its performance in slightly modified environments. The impact of hyperparameters
on generalization is also examined. The objective is to identify the
strengths and limitations of DRL agents in adapting learned policies to similar but
altered environments.

<div style="text-align: center;">
  <table style="border-collapse: collapse; display: inline-block;">
    <tr style="border: none;">
      <td style="border: none; text-align: center; padding: 10px;">
        <img src="Custom_Env/acrobot.gif" alt="Acrobot Animation" width="200" height="150" />
        <div><strong>Acrobot</strong></div>
      </td>
      <td style="border: none; text-align: center; padding: 10px;">
        <img src="Custom_Env/cart_pole.gif" alt="CartPole Animation" width="200" height="150"/>
        <div><strong>CartPole</strong></div>
      </td>
      <td style="border: none; text-align: center; padding: 10px;">
        <img src="Custom_Env/mountain_car.gif" alt="Mountain Car Continuous Animation" width="200" height="150"/>
        <div><strong>Mountain Car Continuous</strong></div>
      </td>
    </tr>
    <tr style="border: none;">
      <td style="border: none; text-align: center; padding: 10px;">
        <img src="Custom_Env/pendulum.gif" alt="Pendulum Animation" width="200" height="150" />
        <div><strong>Pendulum</strong></div>
      </td>
      <td style="border: none; text-align: center; padding: 10px;">
        <img src="Custom_Env/pick_and_place-env.gif" alt="Pick and Place Animation" width="200" height="150"/>
        <div><strong>Pick and Place</strong></div>
      </td>
      <td style="border: none; text-align: center; padding: 10px;"></td>
    </tr>
  </table>
</div>


<br>

>[!NOTE]
>The purpose of this README is to provide instructions and details necessary to replicate the experiments. For more comprehensive information, please consult the [Documentation](./Documentation/) directory.

## Project Structure

### [Algorithms](./Algorithms/)
Contains the implementation of Proximal Policy Optimization (PPO) applied to the different environments. Each environment has a dedicated directory following a consistent structure to ensure uniformity. These directories contain the necessary scripts and configuration files for setting up, training, and testing the models.
```bash
├── Algorithms/
│   ├── PPO_Acrobot/
│   │   ├── Experiments_Config/
│   │   │   ├── ...                # Configurations for training experiments (.yaml)
│   │   ├── Results/
│   │   │   ├── ...                # Results from training experiments (automatically)
│   │   ├── Generalization/
│   │   │   ├── ...                # Generalization experiments and results
│   │   ├── dummy_example.py       # Dummy script to test the environment setup
│   │   ├── Network_Acrobot.py     # Defines Actor-Critic networks
│   │   ├── test_Acrobot.py        # Script to test the trained models
│   │   ├── train_Acrobot.py       # Configures experiments and calls the trainer
│   │   ├── trainer_Acrobot.py     # Implements training of the model using PPO
│   ├── PPO_CartPole/              # Same structure as PPO_Acrobot
│   ├── PPO_ContinuousMountainCar/ # Same structure as PPO_Acrobot
│   ├── PPO_Pendulum/              # Same structure as PPO_Acrobot
│   ├── PPO_PickAndPlace/          # Same structure as PPO_Acrobot
```

### [Cluster](./Cluster/)
Directory related to the experiment scripts (.sh) executed on the UPF Cluster for the Pick and Place scenario, organized into folders based on different hyperparameters. Additionally, it includes the SLURM output files generated from each experiment.
```bash
├── Cluster/
│ ├── BatchSize_exp/        
│ │    ├── ...              # Experiments relateds to the Batch Size
│ ├── LearningRate_exp/         
│ │    ├── ...              # Experiments relateds to the Learning Rates
│ ├── slurms-outputs/...    
│ │    ├── ...              # Slurms obtained in each of the experiments
│ ├── Timesteps_exp/...
│ │    ├── ...              # Experiments relateds to the Timesteps
```

### [Custom Environments](./Custom_Env/)
This directory contains all the custom environments created for the generalization experiments.
```bash
├── Custom_Env/
│   ├── custom_env.py             # Script containing all custom environments
│   ├── dummy_test_custom_env.py  # Dummy script for testing the custom environments
```

## Installation

To run this project, you will need to install the required dependencies. It is recommended to use a virtual environment to manage these dependencies.

1. Clone the repository:

    ```bash
    git clone https://github.com/ialexmp/DRL-Generalization.git
    cd DRL-Generalization
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

 ```bash
cd DRL-Generalization
 ```

### Dummy Experiments

Before starting training and generalization experiments, you can execute various dummy experiments to ensure that everything is set up successfully.

```bash
python ./Algorithms/PPO_[env_name]/dummy_example.py dummy_exp.yaml
```

### Training Experiments

To run training experiments with different configurations, use the experiment script:

```bash
python ./Algorithms/PPO_[env_name]/train.py train_config.yaml
```
You can adjust the training parameters by editing the configuration file located at Experiment_Config/train_config.yaml. Note that you can create different experiments within the same .yaml file, and the results will be automatically organized into separate folders named after each experiment within the .yaml file.

### Model Evaluation 

Once the agent is trained, you can evaluate the performance of the agent using:
```bash 
python ./Algorithms/PPO_[env_name]/test.py [path_from_Results_folder]
# Example : 
# python ./Algorithms/PPO_Acrobot/test.py 1M_lr0_01\exp1_2024-06.07_20-04-46
```
###  Generealization Experiments
After training, you can evaluate the generalization performance of a trained agent in a custom environment with slight modifications using the command:
```bash 
python ./Algorithms/PPO_[env_name]/Generalization/[Experiment_folder]/ZSG_[Experiment_name].py [path_from_Results_folder]
# Example : 
# python ./Algorithms/PPO_Acrobot/Generalziation/Link_Mass/ZSG_linkMass.py 1M_lr0_01\exp1_2024-06.07_20-04-46
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's style guidelines and includes relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
