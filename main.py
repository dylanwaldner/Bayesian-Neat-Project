import os
import sys
sys.path.insert(0, '/scratch/cluster/dylantw/Risto/init/modular_codebase')

# Print the current working directory and sys.path
print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)
print("RIGHT DIRECTORY")

# Add the current directory to sys.path explicitly
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import ray
import json
import torch
import re

import bnn.bayesnn
print(f"BayesNN loaded from: {bnn.bayesnn.__file__}")
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

import bnn_neat
from bnn_neat.genome import DefaultGenome

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = "gpt-4o-mini"



if __name__ == "__main__":
    max_tokens = 10240
    temperature = 1.2
    top_p = 0.95
    danger = 10

    pyro.enable_validation(False)


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
    ground_truth_label_list = []
    ethical_ground_truths = []
    gen_loss_history = []
    gen_ethical_history = []

    num_gens = 2

    global_counter = 0

    # Call the loop logic directly without Gradio
    result, gen_loss_history, gen_ethical_history, ethical_ground_truths, survival_ground_truths = generational_driver(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, strong_bnn, config, num_gens, neat_trainer, global_counter)

    print("Experiment complete. Results saved.")

