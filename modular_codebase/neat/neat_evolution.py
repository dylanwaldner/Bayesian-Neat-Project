import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
import ray
import traceback

# Initialize Ray at the beginning of your script
import numpy as np
from math import ceil
import torch
import torch.nn as nn
import time
import random
import pytest
import sys
from bnn.bayesnn import BayesianNN
from joblib import Parallel, delayed
import torch.multiprocessing as mp

import bnn_neat
#from bnn_neat.checkpoint import Checkpointer
from bnn_neat.config import Config
from bnn_neat.genome import DefaultGenome
from bnn_neat.reproduction import DefaultReproduction
from bnn_neat.species import DefaultSpeciesSet
from bnn_neat.stagnation import DefaultStagnation
from bnn_neat.genes import DefaultConnectionGene, DefaultNodeGene
from bnn_neat.reporting import ReporterSet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesianGenome(DefaultGenome):
    # Extend DefaultGenome to handle weight_mu and weight_sigma
    def __init__(self, key):
        super().__init__(key)
        # Additional initialization if needed

    # Override methods if necessary to handle weight_mu and weight_sigma

def compute_elbo_loss(model_instance, bnn_history, ground_truth_labels, current_index=None, num_samples=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the model is in evaluation mode
    model_instance.eval()

    # Update the model's internal state
    if len(bnn_history) > model_instance.last_update_index:
        model_instance.update_matrix(bnn_history)

    model_instance.current_index = len(bnn_history) - 1 if current_index is None else current_index
    model_instance.bnn_history = bnn_history

    # Prepare dummy x_data and move ground_truth_labels to device
    x_data = torch.empty((1, model_instance.input_size), device=device).fill_(-1)
    y_data = ground_truth_labels.to(device)

    # Use Pyro's Trace_ELBO to compute the loss
    elbo = pyro.infer.Trace_ELBO()
    loss = elbo.loss(model_instance.model, model_instance.guide, x_data, y_data)

    return loss

def evaluate_genome(genome, config, bnn, bnn_history, ground_truth_label_list, all_choices_ethics, device):
    # Reset the BNN's input matrix and last update index
    bnn.input_matrix = None
    bnn.last_update_index = 0

    total_loss = 0.0
    num_entries = 0

    # Initialize lists to record decisions and ethical scores
    decision_history = []
    ethical_score_history = []

    # Create mappings from id to ground truth labels and ethical scores
    ground_truth_label_map = {k: v for d in ground_truth_label_list for k, v in d.items()}
    ethical_score_map = {k: v for d in all_choices_ethics for k, v in d.items()}

    # Get entries where the agent is 'Storyteller' along with their indices
    storyteller_entries = [(idx, entry) for idx, entry in enumerate(bnn_history) if entry['agent'] == 'Storyteller']

    print("Number of Storyteller Entries:", len(storyteller_entries))
    print("Number of Ground Truth Labels:", len(ground_truth_label_map))
    print("Number of Ethical Score Entries:", len(ethical_score_map))

    if not storyteller_entries:
        # No valid entries, set fitness to a default low value
        genome.fitness = -float('inf')
        return genome.fitness

    # Loop over storyteller entries
    for idx, entry in storyteller_entries:
        entry_id = entry.get('id')  # Retrieve the 'id' from the bnn_history entry

        # Retrieve the ground truth labels and ethical scores using the 'id'
        expected_output = ground_truth_label_map.get(entry_id)
        ethical_scores = ethical_score_map.get(entry_id)

        if expected_output is None or ethical_scores is None:
            print(f"Warning: No ground truth data found for id {entry_id}")
            continue  # Skip this entry if data is missing

        # Prepare the expected output tensor
        expected_output_tensor = torch.tensor(expected_output, dtype=torch.float32, device=device)
        expected_output_tensor = expected_output_tensor.unsqueeze(0)  # Add batch dimension if necessary

        # Compute ELBO loss for the current time step without optimization
        loss = compute_elbo_loss(
            bnn,
            bnn_history[:idx + 1],  # Pass the history up to the current index
            expected_output_tensor,
            current_index=idx,
            device=device
        )
        total_loss += loss
        num_entries += 1

        # Perform a forward pass to get predictions
        with torch.no_grad():
            logits = bnn.forward(bnn_history[:idx + 1], current_index=idx, device=device)
            probabilities = torch.sigmoid(logits)

        # Record the decision made by the genome at this time step
        chosen_action = torch.argmax(probabilities).item()
        decision_history.append(chosen_action)

        # Get the ethical score corresponding to the chosen action
        ethical_score = ethical_scores[chosen_action]
        ethical_score_history.append(ethical_score)

    if num_entries > 0:
        average_loss = total_loss / num_entries
    else:
        average_loss = float('inf')  # Set a high loss if no valid entries

    fitness = -average_loss  # Assuming lower loss is better

    # Assign fitness to the genome
    genome.fitness = fitness

    # Attach the decision and ethical score histories to the genome
    genome.decision_history = decision_history
    genome.ethical_score_history = ethical_score_history

    return fitness

@ray.remote(num_gpus=0.2)
def evaluate_genome_remote(genome_id, genome, config, bnn_history, ground_truth_labels, attention_layers, ethical_ground_truths):
    device = torch.device('cuda')
    device_id = torch.cuda.current_device()
    print(f"Genome {genome_id} is using GPU {device_id}")

    try:
        start_time = time.time()

        bnn_history_device = []
        for entry in bnn_history:
            entry_device = {}
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry_device[key] = value.to(device)
                elif isinstance(value, (list, np.ndarray)):
                    entry_device[key] = torch.tensor(value, device=device)
                else:
                    entry_device[key] = value
            bnn_history_device.append(entry_device)

        # Initialize model
        bnn = BayesianNN(genome, config, attention_layers=attention_layers).to(device)

        # Log GPU memory usage after model initialization
        allocated_memory = torch.cuda.memory_allocated(device)
        max_allocated_memory = torch.cuda.max_memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        max_reserved_memory = torch.cuda.max_memory_reserved(device)
        print(f"Genome {genome_id} - Memory Allocated: {allocated_memory / (1024**2):.2f} MB")
        print(f"Genome {genome_id} - Max Memory Allocated: {max_allocated_memory / (1024**2):.2f} MB")
        print(f"Genome {genome_id} - Memory Reserved: {reserved_memory / (1024**2):.2f} MB")
        print(f"Genome {genome_id} - Max Memory Reserved: {max_reserved_memory / (1024**2):.2f} MB")

        # Evaluate fitness
        fitness = evaluate_genome(
            genome, config, bnn, bnn_history_device, ground_truth_labels, ethical_ground_truths, device
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Genome {genome_id} - Evaluation Time: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Exception in genome {genome_id}: {e}")
        traceback.print_exc()
        fitness = float('-inf')

    # Clean up
    del bnn
    torch.cuda.empty_cache()

    return genome_id, fitness

class NeatEvolution:
    def __init__(self, config, config_path, bnn):
        self.config = config
        self.config_path = config_path
        self.bnn = bnn
        self.stagnation_limit = 5  # Number of generations to wait for improvement

        self.population_tradeoffs = []
        
        # Initialize stagnation tracking variables
        self.best_fitness = None
        self.generations_without_improvement = 0

        # Extract BNN parameters
        num_inputs, num_outputs, connections, attention_layers = self.extract_bnn_parameters(bnn)

        self.attention_layers = attention_layers

        # Update NEAT configuration
        self.update_neat_config(self.config, num_inputs, num_outputs)

        # Extract optimized parameters from BNN
        optimized_params = bnn.get_optimized_parameters()

        # Create initial population based on BNN
        initial_population = self.create_initial_population(connections, optimized_params)

        # Initialize NEAT population without initial_state
        self.population = bnn_neat.Population(self.config)

        # Replace the randomly initialized population with your custom population
        self.population.population = initial_population

        # Speciate the population
        self.population.species.speciate(self.config, self.population.population, self.population.generation)

        # Add reporters
        self.population.add_reporter(bnn_neat.StdOutReporter(True))
        self.stats = bnn_neat.StatisticsReporter()
        #checkpointer = Checkpointer(generation_interval=5, time_interval_seconds=600)

        self.population.add_reporter(self.stats)

    def update_neat_config(self, config, num_inputs, num_outputs):
        """
        Updates the NEAT configuration to match the BNN's input and output sizes.
        """
        config.genome_config.num_inputs = num_inputs
        config.genome_config.num_outputs = num_outputs
        #config.genome_config.initial_connection = 'full'  # Adjust as needed


    def extract_bnn_parameters(self, bnn):
        """
        Extracts the necessary parameters from the BayesianNN object.

        Returns:
            num_inputs (int): Number of input nodes.
            num_outputs (int): Number of output nodes.
            hidden_layers (list): List containing the number of nodes in each hidden layer.
            connections (dict): Dictionary of connections with their properties.
        """
        # Extract the number of input and output nodes
        num_inputs = bnn.config.genome_config.num_inputs
        num_outputs = bnn.config.genome_config.num_outputs
        # Extract hidden layer configurations
        # Assuming bnn.hidden_layers is a list like [64, 32] for two hidden layers
        #hidden_layers = bnn.config.genome_config.hidden_layers MIGHT have to come back to this? I cant tell
        hidden_layers = []

        # Extract connections
        connections = bnn.get_connections()

        # Extract attention layers (query, key, value projections)
        attention_layers = {
            'query_proj': bnn.query_proj.state_dict(),
            'key_proj': bnn.key_proj.state_dict(),
            'value_proj': bnn.value_proj.state_dict()
        }

        return num_inputs, num_outputs, connections, attention_layers

    def create_initial_population(self, connections, optimized_params):
        """
        Creates the initial NEAT population based on the BNN's connections.

        Args:
            connections (dict): Connections from the BNN.

        Returns:
            population (dict): A dictionary of genomes for NEAT's initial population.
        """
        population = {}
        for i in range(self.config.pop_size):
            genome_id = i
            genome = self.create_genome_from_bnn(genome_id, connections, optimized_params)
            population[genome_id] = genome
        return population

    def create_genome_from_bnn(self, genome_id, connections, optimized_params):
        """
        Creates a NEAT genome initialized with BNN's connections.

        Args:
            genome_id (int): The ID of the genome.
            connections (dict): BNN's connections.

        Returns:
            genome: A NEAT genome object.
        """
        genome = BayesianGenome(genome_id)
        genome.configure_new(self.config.genome_config)

        # Initialize genome's connections based on BNN's connections and optimized parameters
        for conn_key, conn_data in connections.items():
            conn_gene = DefaultConnectionGene(conn_key)
            weight_mu_name = f"w_mu_{conn_key}"
            weight_sigma_name = f"w_sigma_{conn_key}"
            if weight_mu_name in optimized_params and weight_sigma_name in optimized_params:
                # For connections
                conn_gene.weight_mu = optimized_params[weight_mu_name].cpu().squeeze().item()
                conn_gene.weight_sigma = optimized_params[weight_sigma_name].cpu().squeeze().item()
                #print(f"NEAT Init {conn_key} - Mu: {conn_gene.weight_mu}, Sigma: {conn_gene.weight_sigma}")
            else:
                conn_gene.weight_mu = conn_data['weight_mu']
                conn_gene.weight_sigma = conn_data['weight_sigma']
            conn_gene.enabled = conn_data['enabled']
            genome.connections[conn_key] = conn_gene

        # Similarly initialize node biases
        for node_id, node_data in self.bnn.nodes.items():
            if node_id < 0:
                continue  # Skip input nodes
            node_gene = genome.nodes.get(node_id)
            if node_gene is None:
                node_gene = DefaultNodeGene(node_id)
                genome.nodes[node_id] = node_gene
            bias_mu_name = f"b_mu_{node_id}"
            bias_sigma_name = f"b_sigma_{node_id}"
            if bias_mu_name in optimized_params and bias_sigma_name in optimized_params:  
                node_gene.bias_mu = optimized_params[bias_mu_name].cpu().squeeze().item()
                node_gene.bias_sigma = optimized_params[bias_sigma_name].cpu().squeeze().item()
            else:
                node_gene.bias_mu = node_data['bias_mu']
                node_gene.bias_sigma = node_data['bias_sigma']

        return genome

    def run_neat_step(self, strong_bnn, bnn_history, ground_truth_labels, ethical_ground_truths):
        winner = self.population.run(
            lambda genomes, _, k: self.fitness_function(genomes, self.config, k, bnn_history, ground_truth_labels, ethical_ground_truths),
            n=1
        )
        return winner

    def fitness_function(self, genomes, config, k, bnn_history, ground_truth_labels, ethical_ground_truths):
        attention_layers = self.attention_layers

        # Place shared data in Ray object store
        bnn_history_ref = ray.put(bnn_history)
        ground_truth_labels_ref = ray.put(ground_truth_labels)
        attention_layers_ref = ray.put(attention_layers)
        ethical_ground_truths_ref = ray.put(ethical_ground_truths)
        #print("Type of bnn_history_ref (Fitness Function):", type(bnn_history_ref))
        #print("Type of ground_truth_labels_ref (Fitness Function):", type(ground_truth_labels_ref))
        #print("Type of attention_layers_ref (Fitness Function):", type(attention_layers_ref))


        # Create a mapping from genome IDs to the original genome objects
        genome_dict = dict(genomes)

        # Launch evaluation tasks in parallel using Ray
        futures = [
            evaluate_genome_remote.remote(
                genome_id, genome, config, bnn_history_ref, ground_truth_labels_ref, attention_layers_ref, ethical_ground_truths_ref
            )
            for genome_id, genome in genomes
        ]

        # Retrieve results
        results = ray.get(futures)

        # Update genomes and collect fitness scores
        fitness_report = []
        decision_histories = []
        for genome_id, fitness in results:
            genome = genome_dict[genome_id]
            genome.fitness = fitness
            fitness_report.append(fitness)

            # Collect the decision and ethical histories
            decision_histories.append({
                'genome_id': genome_id,
                'decisions': genome.decision_history,
                'ethical_scores': genome.ethical_score_history,
                'fitness': fitness
            })

        # Store the collected data for this generation
        self.population_tradeoffs.append({
            'generation': k,
            'tradeoffs': decision_histories
        })

        # Calculate summary statistics
        mean_fitness = np.mean(fitness_report)
        median_fitness = np.median(fitness_report)
        std_fitness = np.std(fitness_report)
        upper_q = np.percentile(fitness_report, 75)
        lower_q = np.percentile(fitness_report, 25)
        iqr_fitness = upper_q - lower_q

        # Get the top 5 and bottom 5 fitness scores
        sorted_fitness = sorted(fitness_report, reverse=True)
        top_5_fitness = sorted_fitness[:5]
        bottom_5_fitness = sorted_fitness[-5:]

        # Print comprehensive fitness summary
        print(f"Generation {k} Fitness Summary:")
        print(f"  Mean Fitness: {mean_fitness:.4f}")
        print(f"  Median Fitness: {median_fitness:.4f}")
        print(f"  Standard Deviation: {std_fitness:.4f}")
        print(f"  Upper Quartile: {upper_q:.4f}")
        print(f"  Lower Quartile: {lower_q:.4f}")
        print(f"  Interquartile Range (IQR): {iqr_fitness:.4f}")
        print(f"  Top 5 Fitness Scores: {top_5_fitness}")
        print(f"  Bottom 5 Fitness Scores: {bottom_5_fitness}")

        # Find current best fitness
        current_best_fitness = max(fitness_report)

        # Check for fitness improvement
        if self.best_fitness is None or current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.generations_without_improvement = 0
            print(f"New best fitness: {self.best_fitness:.4f}")
        else:
            self.generations_without_improvement += 1
            print(f"No improvement in fitness for {self.generations_without_improvement} generations")

        # Check for stagnation
        if self.generations_without_improvement >= self.stagnation_limit:
            print(f"Stopping evolution: No improvement in fitness for {self.stagnation_limit} generations.")
            return True  # This will now cause Population.run() to break the loop
        else:
            return False  # Continue evolution

