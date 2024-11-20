# test_bayesian_nn.py

import os
import numpy as np
import torch
import pyro.distributions as dist

from bnn import BayesianNN
from evolve_neat import NeatEvolution, BayesianGenome
# Add the path to the neat directory
import sys
sys.path.insert(0, '/scratch/cluster/dylantw/Risto/init/bnn-neat-python')
print("sys.path:", sys.path)
import bnn_neat
print("neat module location:", bnn_neat.__file__)
from bnn_neat.genome import DefaultGenome, DefaultGenomeConfig

def get_test_config():
    # Create a NEAT configuration object
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-2')

    # Write a minimal configuration to 'config-feedforward' for testing purposes
    with open(config_path, 'w') as f:
        f.write("""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1.0
pop_size              = 5
reset_on_extinction   = False

[BayesianGenome]
activation_default      = relu
activation_mutate_rate  = 0.1
activation_options      = relu sigmoid tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# Bias configuration
bias_mu_init_mean       = 0.0
bias_mu_init_stdev      = 0.5
bias_sigma_init_mean    = 0.5
bias_sigma_init_stdev   = 0.1
bias_mu_mutate_power    = 0.1
bias_mu_mutate_rate     = 0.5
bias_mu_replace_rate    = 0.05
bias_sigma_mutate_power = 0.05
bias_sigma_mutate_rate  = 0.5
bias_sigma_replace_rate = 0.05
bias_mu_max_value       = 10.0
bias_mu_min_value       = -10.0
bias_sigma_max_value    = 2.0
bias_sigma_min_value    = 0.01

# Weight configuration
weight_mu_init_mean       = 0.0
weight_mu_init_stdev      = 0.5
weight_sigma_init_mean    = 0.5
weight_sigma_init_stdev   = 0.1
weight_mu_mutate_power    = 0.1
weight_mu_mutate_rate     = 0.5
weight_mu_replace_rate    = 0.05
weight_sigma_mutate_power = 0.05
weight_sigma_mutate_rate  = 0.5
weight_sigma_replace_rate = 0.05
weight_mu_max_value       = 10.0
weight_mu_min_value       = -10.0
weight_sigma_max_value    = 2.0
weight_sigma_min_value    = 0.01

# Response configuration
response_mu_init_mean       = 1.0
response_mu_init_stdev      = 0.5
response_sigma_init_mean    = 0.1
response_sigma_init_stdev   = 0.05
response_mu_mutate_power    = 0.1
response_mu_mutate_rate     = 0.5
response_mu_replace_rate    = 0.05
response_sigma_mutate_power = 0.05
response_sigma_mutate_rate  = 0.5
response_sigma_replace_rate = 0.05
response_mu_max_value       = 5.0
response_mu_min_value       = -5.0
response_sigma_max_value    = 2.0
response_sigma_min_value    = 0.01

# Node compatibility settings
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# Connection mutation rates
conn_add_prob           = 0.3
conn_delete_prob        = 0.2

# Number of inputs, outputs, and connections
num_hidden              = 0
num_inputs              = 3
num_outputs             = 2
initial_connection      = full

feed_forward            = True

enabled_default         = True
enabled_mutate_rate     = 0.1

node_add_prob           = 0.2
node_delete_prob        = 0.1



[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 1
survival_threshold = 0.2
""")

    # Load the configuration
    config = bnn_neat.Config(BayesianGenome, bnn_neat.DefaultReproduction,
                         bnn_neat.DefaultSpeciesSet, bnn_neat.DefaultStagnation,
                         config_path)
    return config

def create_sample_genome(config):
    # Create a genome with a unique ID
    genome_id = 0
    genome = BayesianGenome(genome_id)
    genome.configure_new(config.genome_config)

    # Manually set some connections and weights for testing
    # Create connections from each input to each output
    for input_node_id in range(-config.genome_config.num_inputs, 0):
        for output_node_id in range(config.genome_config.num_outputs):
            key = (input_node_id, output_node_id)
            conn_gene = bnn_neat.DefaultConnectionGene(key)
            # Assign random weights and sigmas
            conn_gene.weight_mu = np.random.randn()
            conn_gene.weight_sigma = np.abs(np.random.randn())
            conn_gene.enabled = True
            genome.connections[key] = conn_gene

    # Return the genome
    return genome

def test_bayesian_nn():
    config = get_test_config()
    genome = create_sample_genome(config)

    # Print genome details
    print("Genome connections:")
    for key, conn in genome.connections.items():
        print(f"  {key}: weight_mu={conn.weight_mu}, weight_sigma={conn.weight_sigma}, enabled={conn.enabled}")

    # Initialize the BayesianNN with the genome and config
    bnn = BayesianNN(genome, config)

    # Print BNN connections
    print("\nBNN connections:")
    for key, conn_data in bnn.get_connections().items():
        print(f"  {key}: {conn_data}")

    # Verify that the weight matrices are correctly constructed
    print("\nWeight mu matrix:")
    print(bnn.weight_mu_matrix)

    print("\nWeight sigma matrix:")
    print(bnn.weight_sigma_matrix)

    # Create sample input data
    sample_input = np.array([0.5, -1.2, 3.3])

    # Run the forward pass
    output = bnn.forward(sample_input)
    print("\nForward pass output:")
    print(output.detach().numpy())

if __name__ == '__main__':
    test_bayesian_nn()

