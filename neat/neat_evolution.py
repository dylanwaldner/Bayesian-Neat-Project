import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
import random

import traceback

global rank, local_rank

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()

    num_gpus_per_node = 3  # Adjust to your system
    gpu_id = local_rank % num_gpus_per_node
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"MPI Initialized. GPU ID: {gpu_id}")
except ImportError:
    # Dummy MPI setup
    class DummyComm:
        def bcast(self, data, root):
            return data  # No-op broadcast for single-process

        def gather(self, data, root):
            # In a single-process setup, just return a list containing the data
            return [data]

        def Get_rank(self):
            return 0  # Simulate rank 0

        def Get_size(self):
            return 1  # Simulate a single process

    comm = DummyComm()
    rank = 0
    size = 1
    local_rank = 0
    num_gpus_per_node = 1
    gpu_id = local_rank % num_gpus_per_node
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"MPI not available. Simulated GPU ID: {gpu_id}")

# Initialize Ray at the beginning of your script
import numpy as np
from math import ceil

import torch
print("Number of GPUs available:", torch.cuda.device_count())
import torch.nn as nn

import time
import random
import pytest
import sys
from neat.neat_utils import save_evolution_results
from bnn.bayesnn import BayesianNN
import json
import pyro
import pyro.infer

import bnn_neat
#from bnn_neat.checkpoint import Checkpointer
from bnn_neat.config import Config
from bnn_neat.genome import DefaultGenome
from bnn_neat.reproduction import DefaultReproduction
from bnn_neat.species import DefaultSpeciesSet
from bnn_neat.stagnation import DefaultStagnation
from bnn_neat.genes import DefaultConnectionGene, DefaultNodeGene
from bnn_neat.reporting import ReporterSet, StdOutReporter, BaseReporter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesianGenome(DefaultGenome):
    # Extend DefaultGenome to handle weight_mu and weight_sigma
    def __init__(self, key):
        super().__init__(key)
        # Additional initialization if needed

    # Override methods if necessary to handle weight_mu and weight_sigma

def compute_bce_loss(model_instance, bnn_history, ground_truth_labels, current_index=None, device=None):
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

    # Perform a forward pass to compute logits
    logits = model_instance.forward_bce(x_data, device=device)

    # Apply sigmoid activation to get probabilities
    probabilities = torch.sigmoid(logits)

    # Compute BCE loss
    loss_fn = torch.nn.BCELoss()
    loss = loss_fn(probabilities, y_data)

    model_instance.current_index = None

    return loss.item(), logits, probabilities


def evaluate_genome(genome, config, bnn, bnn_history, ground_truth_label_list, all_choices_ethics, device):
    print(f"Evaluating genome: {genome.key}")
    print(f"BNN History Length: {len(bnn_history)}")
    print(f"Number of Ground Truth Label Entries: {len(ground_truth_label_list)}")
    print(f"Number of Ethical Score Entries: {len(all_choices_ethics)}")

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

    print(f"Ground Truth Label Map Keys: {list(ground_truth_label_map.keys())[:5]}")
    print(f"Ethical Score Map Keys: {list(ethical_score_map.keys())[:5]}")

    # Get entries where the agent is 'Storyteller' along with their indices
    storyteller_entries = [(idx, entry) for idx, entry in enumerate(bnn_history) if entry['agent'] == 'Storyteller']

    print("Number of Storyteller Entries:", len(storyteller_entries))
    print("Number of Ground Truth Labels:", len(ground_truth_label_map))
    print("Number of Ethical Score Entries:", len(ethical_score_map))

    if not storyteller_entries:
        # No valid entries, set fitness to a default low value
        genome.fitness = -float('inf')
        print("No storyteller entries found. Assigning minimum fitness.")
        return genome.fitness
    
    evaluation_window = genome.evaluation_window

    # Randomly select a subset of storyteller entries if there are enough entries
    if len(storyteller_entries) > evaluation_window:
        selected_storyteller_entries = random.sample(storyteller_entries, evaluation_window)
    else:
        # If fewer entries exist, use all of them
        selected_storyteller_entries = storyteller_entries

    # Loop over storyteller entries
    for idx, entry in selected_storyteller_entries:
        print(f"Selected Entry IDs: {[entry.get('id') for _, entry in selected_storyteller_entries]}")

        entry_id = entry.get('id')  # Retrieve the 'id' from the bnn_history entry

        # Retrieve the ground truth labels and ethical scores using the 'id'
        expected_output = ground_truth_label_map.get(entry_id)
        ethical_scores = ethical_score_map.get(entry_id)

        if expected_output is None or ethical_scores is None:
            print(f"Warning: No ground truth data found for id {entry_id}")
            continue  # Skip this entry if data is missing

        # Prepare the expected output tensor
        expected_output_tensor = torch.tensor(expected_output, dtype=torch.float32, device=device)

        # Add batch dimension if necessary
        if len(expected_output_tensor.shape) == 1:  # Check if it lacks a batch dimension
            expected_output_tensor = expected_output_tensor.unsqueeze(0)

        # Compute BCE loss and retrieve predictions
        loss, logits, probabilities = compute_bce_loss(
            bnn,
            bnn_history[:idx + 1],  # Pass the history up to the current index
            expected_output_tensor,
            current_index=idx,
            device=device
        )
        total_loss += loss
        num_entries += 1

        # Record the decision made by the genome at this time step
        chosen_action = torch.argmax(probabilities).item()
        decision_history.append((idx, chosen_action))

        # Get the ethical score corresponding to the chosen action
        ethical_score = ethical_scores[chosen_action]
        ethical_score_history.append((idx, ethical_score))

    print("Num Entries (in evaluate_genome()): ", num_entries)
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

    print(f"Average Loss: {average_loss}")
    print(f"Genome Fitness: {fitness}")
    print(f"Decision History: {decision_history[:5]}")
    print(f"Ethical Score History: {ethical_score_history[:5]}")

    return fitness

def evaluate_genome_remote(genome_id, genome, config, bnn_history, ground_truth_labels, attention_layers, ethical_ground_truths):
    global rank, local_rank  # Access global variables

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Genome {genome_id} - Rank {rank}, Local Rank {local_rank}, using device {device}")

    if torch.cuda.is_available():
        print(f"Genome {genome_id} - Rank {rank}, Local Rank {local_rank} - Visible GPUs: {torch.cuda.device_count()}")
        print(f"Genome {genome_id} - Active CUDA Device: {torch.cuda.current_device()}")
        print(f"Genome {genome_id} - CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print(f"Genome {genome_id} - No CUDA device available. Falling back to CPU.")

    try:
        start_time = time.time()
        # Move bnn_history to the assigned device
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

        # Initialize model on the assigned device
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
        comm = None
        rank = 0  # Simulate being the first process
        size = 1  # Simulate a single process

        # Simulate a single GPU setup
        local_rank = 0
        num_gpus_per_node = 1  # Set to the number of GPUs you want to simulate
        gpu_id = local_rank % num_gpus_per_node
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
    def __init__(self, config, config_path, bnn, neat_iteration=None):
        self.config = config
        self.config_path = config_path
        self.bnn = bnn
        self.stagnation_limit = 5  # Number of generations to wait for improvement
        self.neat_iteration = neat_iteration

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
        self.max_generations = 1
        winner = self.population.run(
            lambda genomes, _, k: self.fitness_function(genomes, self.config, k, bnn_history, ground_truth_labels, ethical_ground_truths),
            n=self.max_generations,
            neat_iteration=self.neat_iteration
        )
        return winner

    def fitness_function(self, genomes, config, k, bnn_history, ground_truth_labels, ethical_ground_truths):
        attention_layers = self.attention_layers

        # Initialize evolution results if not already defined
        if not hasattr(self, "evolution_results"):
            self.evolution_results = {
                "fitness_summary": [],  # Holds per-generation fitness statistics
                "population_tradeoffs": []  # Stores decision and ethical histories
            }

        # Broadcast necessary data to all processes
        if rank == 0:
            data = (bnn_history, ground_truth_labels, attention_layers, ethical_ground_truths)
        else:
            data = None

        if comm is not None:
            bnn_history, ground_truth_labels, attention_layers, ethical_ground_truths = comm.bcast(data, root=0)
        else:
            # Single-process case: Use `data` directly
            bnn_history, ground_truth_labels, attention_layers, ethical_ground_truths = data

        # Convert genomes to list and create genome_dict
        genomes = list(genomes)
        genome_dict = dict(genomes)

        # Distribute genomes among processes
        num_genomes = len(genomes)
        genomes_per_process = num_genomes // size
        remainder = num_genomes % size

        start_index = rank * genomes_per_process + min(rank, remainder)
        end_index = start_index + genomes_per_process + (1 if rank < remainder else 0)

        local_genomes = genomes[start_index:end_index]

        # Evaluate local genomes
        local_results = []

        # Determine the top-performing genomes if needed
        if k > 5:
            sorted_genomes = sorted(local_genomes, key=lambda g: g[1].fitness, reverse=True)
            top_percentage = 0.2  # Top 20%
            top_count = int(len(sorted_genomes) * top_percentage)
            top_genome_ids = {g[0] for g in sorted_genomes[:top_count]}  # Set of top genome IDs


        for genome_id, genome in local_genomes:
            if k <= 5:  # Early exploration phase
                genome.evaluation_window = 5
            elif 5 < k <= 15:  # Intermediate phase
                if genome_id in top_genome_ids:
                    genome.evaluation_window = 10
                else:
                    genome.evaluation_window = 5
            elif k > 15:  # Later phase with more refined genomes
                if genome_id in top_genome_ids:
                    genome.evaluation_window = 15
                else:
                    genome.evaluation_window = 5

            genome_id, fitness = evaluate_genome_remote(genome_id, genome, config, bnn_history, ground_truth_labels, attention_layers, ethical_ground_truths)
            local_results.append((genome_id, fitness))

        # Gather results at root process
        all_results = comm.gather(local_results, root=0)

        if rank == 0:
            # Combine results from all processes
            all_results = [item for sublist in all_results for item in sublist]
            # Update genomes and collect fitness scores
            fitness_report = []
            decision_histories = []
            for genome_id, fitness in all_results:
                genome = genome_dict[genome_id]
                fitness_report.append(fitness)

                # Collect the decision and ethical histories
                decision_histories.append({
                    'genome_id': genome_id,
                    'decisions': genome.decision_history,
                    'ethical_scores': genome.ethical_score_history,
                    'fitness': genome.fitness,
                    'mutation_history': genome.mutation_history
                })

            # Store the collected data for this generation
            self.evolution_results["population_tradeoffs"].append({
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

            # Find current best fitness
            current_best_fitness = max(fitness_report)

            # Summarize fitness
            fitness_summary = {
                'generation': k,
                'mean_fitness': mean_fitness,
                'median_fitness': median_fitness,
                'std_fitness': std_fitness,
                'upper_quartile': upper_q,
                'lower_quartile': lower_q,
                'iqr_fitness': iqr_fitness,
                'top_5_fitness': top_5_fitness,
                'bottom_5_fitness': bottom_5_fitness,
                'best_fitness': current_best_fitness
            }
            self.evolution_results["fitness_summary"].append(fitness_summary)

            if k == self.max_generations:
                save_evolution_results(self.evolution_results, self.population_tradeoffs, neat_iteration = self.neat_iteration)

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
                save_evolution_results(self.evolution_results, self.population_tradeoffs, neat_iteration = self.neat_evolution)
                return True  # This will now cause Population.run() to break the loop
            else:
                return False  # Continue evolution

        else:
            continue_evolution = None

        # Broadcast the decision to all processes
        if comm is not None:
            continue_evolution = comm.bcast(continue_evolution, root=0)

        return continue_evolution

