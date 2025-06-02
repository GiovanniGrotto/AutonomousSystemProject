# AutonomousSystemProject

This repository provides a modular framework for developing, training, and evaluating autonomous agents in a custom Overcooked-style environment. The system supports multiple reinforcement learning algorithms and is designed for flexible experimentation.

---

## ✅ Getting Started

### Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/GiovanniGrotto/AutonomousSystemProject.git
cd AutonomousSystemProject
pip install -r requirements.txt
```

_Note: You have to install the Overcooked-AI environment manually or ensure it's properly set up._

### Training

Run one of the training scripts:

```bash
python -m algorithms.ppo
```
Thi will run the spicified algorithm in the specified layouts, saving the run on wandb and the checkpoint into the folder saved_models

### Evaluation

Evaluate a trained agent:

```bash
python -m utils.evaluate_agent
```
This evaluates a given agent in a layout, saving the trajectory as both a video and a series of PNG images, along with the action values or probabilities for each agent at each state in the evaluations folder.

_Note: For both the scripts the parameters as layouts, path to save or agents to use have to be set from the code._

---

## 📽 Demo

Watch `Demo Video.mov` for an example of a trained agent performing in the environment.

---


## 📂 Notes

- Trained models are stored in `algorithms/saved_models/`
- `.env` should contains environment-specific configurations and variables like the wandb API KEY.
