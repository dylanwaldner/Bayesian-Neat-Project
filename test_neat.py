# test_bayesian_neat_large.py
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Add the path to the neat directory
sys.path.insert(0, '/scratch/cluster/dylantw/Risto/init/bnn-neat-python')
import bnn_neat
from bnn_neat.genome import DefaultGenome, DefaultGenomeConfig

# Set random seed for reproducibility
random.seed(42)

# Define large input and output dimensions
num_inputs = 7686
num_outputs = 4

# Generate random input-output pairs for testing
# For this test, generate random binary inputs and outputs
test_inputs = np.random.rand(100, num_inputs)  # 100 random input samples with 7687 features
test_outputs = np.random.randint(2, size=(100, num_outputs))  # Binary outputs with 4 output neurons

def eval_genome(genome, config):
    """Evaluates a single genome's fitness for large input-output dimensions."""
    net = bnn_neat.nn.FeedForwardNetwork.create(genome, config)

    error_sum = 0.0
    uncertainty_sum = 0.0
    for xi, xo in zip(test_inputs, test_outputs):
        output = net.activate(xi)
        error = np.sum((np.array(output) - np.array(xo)) ** 2)  # Mean squared error for all outputs
        # Optional: Add uncertainty (sigma) handling here
        uncertainty_sum += net.get_variance() if hasattr(net, 'get_variance') else 0.0  # Assumes a variance method
        error_sum += error

    # Fitness can combine both performance and uncertainty
    genome.fitness = (1.0 - (error_sum / len(test_inputs))) - uncertainty_sum
    
    # Explicitly print the state of the connections, including the 'enabled' field
#    print(f"Evaluating genome {genome.key} connections:")
 #   for conn_key, conn_gene in genome.connections.items():
  #      print(f"Connection {conn_key}: enabled={conn_gene.enabled}, weight_mu={conn_gene.weight_mu}, weight_sigma={conn_gene.weight_sigma}")

def eval_genomes(genomes, config):
    """Evaluates all genomes in the population."""
    for genome_id, genome in genomes:
        eval_genome(genome, config)

def run_neat():
    # Define the path to the configuration file
    config_file = "config-feedforward"  # Update with your actual path
    config = bnn_neat.Config(bnn_neat.DefaultGenome, bnn_neat.DefaultReproduction,
                         bnn_neat.DefaultSpeciesSet, bnn_neat.DefaultStagnation,
                         config_file)

    print("Config loaded successfully")
    print(config.genome_config.num_inputs)
    print(config.genome_config.num_outputs)

    # Create the population
    p = bnn_neat.Population(config)

    # Add reporters to display progress in the terminal and to collect statistics
    p.add_reporter(bnn_neat.StdOutReporter(True))
    stats = bnn_neat.StatisticsReporter()
    p.add_reporter(stats)

    # Optionally, save checkpoints (uncomment to enable)
    # p.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix='neat-checkpoint-'))

    # Run NEAT for up to 100 generations
    winner = p.run(eval_genomes, n=2)

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    # Visualize fitness over generations
    visualize_fitness(stats)

    # Visualize species evolution
    visualize_species(stats)

    # Show output of the most fit genome against test data
    print('\nOutput:')
   # winner_net = bnn_neat.nn.FeedForwardNetwork.create(winner, config)
    #for xi, xo in zip(test_inputs[:5], test_outputs[:5]):  # Show results for first 5 test cases
     #   output = winner_net.activate(xi)
      #  print(f"Input: {xi[:5]}..., Expected Output: {xo}, Network Output: {output}")

def visualize_fitness(stats):
    """Visualize fitness across generations."""
    generation = range(len(stats.most_fit_genomes))
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    
    plt.plot(generation, best_fitness, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_species(stats):
    """Visualize speciation over generations."""
    species_sizes = stats.get_species_sizes()
    plt.plot(species_sizes)
    plt.xlabel("Generation")
    plt.ylabel("Number of Species")
    plt.title("Speciation over Generations")
    plt.show()

if __name__ == '__main__':
    run_neat()

