import os
import sys

# Print the current working directory and sys.path
print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)

# Add the current directory to sys.path explicitly
sys.path.insert(0, os.getcwd())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import ray
import json
import torch
import re

import bnn.bayesnn
from bnn.bayesnn import BayesianNN

import numpy as np
import matplotlib.pyplot as plt
import pyro
import torch.multiprocessing as mp

from loops import main_loop, generational_driver
from utils.plotting import plot_loss_and_survival, plot_survival_and_ethics, plot_loss_and_ethics
from utils.utils_logging import save_experiment_results
from neat.neat_evolution import NeatEvolution
from utils.text_generation import generate_text
from utils.text_utils import normalize_string, trim_response, extract_choices_and_intro
from bnn.bnn_utils import update_bnn_history

import sys
sys.path.insert(0, '/scratch/cluster/dylantw/Risto/init/bnn-neat-python')

import bnn_neat
from bnn_neat.genome import DefaultGenome

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = "gpt-4o-mini"



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    ray.init()

    max_tokens = 10240
    temperature = 1.2
    top_p = 0.95
    danger = 10

    # Set up NEAT configurations
    config_path = "config-feedforward"
    config = bnn_neat.config.Config(bnn_neat.DefaultGenome, bnn_neat.DefaultReproduction,
                                bnn_neat.DefaultSpeciesSet, bnn_neat.DefaultStagnation,
                                config_path)

    # Create an initial genome for the strong agent
    genome_id = 0
    strong_genome = DefaultGenome(genome_id)
    strong_genome.configure_new(config.genome_config)

    # Initialize the BNN with the genome and config
    strong_bnn = BayesianNN(strong_genome, config)

    neat_trainer = NeatEvolution(config, config_path, strong_bnn)

    # get votes between strong agent and weak agent
    #votes, shared_history, bnn_history = power_division(max_tokens, temperature, top_p)
    votes = {'strong': 0, 'weak': 10} 
    shared_history = []
    bnn_history = []
    num_gens = 4

    global_counter = 0

    # Call the loop logic directly without Gradio
    result, loss, survival, ethics, ethical_ground_truths, survival_ground_truths = generational_driver(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, strong_bnn, config, num_gens, neat_trainer, global_counter)
    print("RESULT: ", result)  # You can save it or print the result
    print("LOSS: ", loss)
    print("SVI DECISION ETHICS: ", ethics)
    print("ETHICAL GROUND TRUTHS: ", ethical_ground_truths)
    print("SURVIVAL HISTORY: ", survival)
    # Calculate the total rounds survived across all games
    total_rounds_survived = sum(survival.values())

    # Calculate the total possible rounds (50 per game)
    total_possible_rounds = 50 * len(survival)

    # Calculate the survival rate as a percentage
    survival_rate = (total_rounds_survived / total_possible_rounds) * 100
    print(f"Survival Rate: {survival_rate:.2f}%")

    # Generate and save the progress plot
    average_loss_per_gen = [sum(l) / len(l) for l in loss]
    survival_counts = list(survival.values())
    average_ethical_score_per_gen = [sum(e) / len(e) if len(e) > 0 else 0 for e in ethics]

    save_experiment_results(result, loss, survival, ethics, ethical_ground_truths, survival_rate)

    # Generate and save individual plots
    plot_loss_and_survival(average_loss_per_gen, survival_counts, filename='loss_and_survival_plot.png')
    plot_survival_and_ethics(survival_counts, average_ethical_score_per_gen, filename='survival_and_ethics_plot.png')
    plot_loss_and_ethics(average_loss_per_gen, average_ethical_score_per_gen, filename='loss_and_ethics_plot.png')

    print("done")
