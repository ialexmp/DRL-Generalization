# Exploring Generalization in Reinforcement Learning for Pick and Place Robotics Tasks

## Introduction

This research investigates the generalization capabilities of reinforcement learning (RL) algorithms in the context of pick and place robotics tasks. The study focuses on assessing how well RL agents trained in simpler environments can adapt to unseen scenarios, a crucial aspect for real-world applications. The research utilizes Gymnasium robotics environments with MuJoCo physics engine to design a set of pick and place tasks of varying complexities. A Proximal Policy Optimization (PPO) algorithm is employed as the RL method for training agents to perform these tasks.

The experimentation involves training RL agents in simpler pick and place environments and evaluating their performance in both seen and unseen scenarios. Through iterative training and evaluation cycles, the study analyzes the agents' ability to generalize across different environments, varying object shapes, sizes, and configurations. Additionally, the impact of hyperparameters and neural network architectures on generalization is explored.

The results highlight the strengths and limitations of RL agents in generalizing learned policies to novel situations. Insights gained from this research contribute to a better understanding of how RL algorithms can be adapted and enhanced for real-world robotics applications. The findings offer valuable implications for the development of adaptive and robust RL systems in diverse pick and place robotics tasks.

<div align="center">
  <img src="Custom_Env/pick_and_place-env.gif" alt="Pick and Place Animation" width="250"/>
</div>

## Project Structure

The repository is organized as follows:
```bash
.
├── Algorithms/
│ ├── PPO/
│ ├── dummy_example.py/
├── Cluster/
│ ├── exploration/
│ ├── experiments/
├── Custom_Env/
│ ├── agents/
│ ├── environments/
│ ├── utils/
├── results/
├── tests/
└── README.md
```

- **data/**: Contains raw and processed datasets.
- **notebooks/**: Jupyter notebooks for data exploration and experiment documentation.
- **src/**: Source code for the project, including agents, environments, and utility functions.
- **results/**: Directory for storing results from experiments.
- **tests/**: Unit tests for the codebase.

## Installation

To run this project, you will need to install the required dependencies. It is recommended to use a virtual environment to manage these dependencies.

1. Clone the repository:

    ```bash
    git clone https://github.com/username/repository.git
    cd repository
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

### Training an RL Agent

To train a reinforcement learning agent, run the following command:

```bash
python src/train.py --config configs/train_config.yaml
```
You can adjust the training parameters by editing the configuration file located in configs/train_config.yaml.

### Running Experiments
To run experiments with different configurations, use the experiment script:
```bash 
python src/evaluate.py --model-path models/agent_model.pth
```
###  Evaluating Results
After training, you can evaluate the performance of the trained agent using:
```bash 
python src/evaluate.py --model-path models/agent_model.pth
```

## Experiments

The experiments conducted in this project are documented in the notebooks/experiments/ directory. Each notebook contains detailed descriptions, code, and results of individual experiments

## Results
Results from the experiments, including performance metrics and trained models, are stored in the results/ directory. Visualizations and analysis of these results are also included in the notebooks.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's style guidelines and includes relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
