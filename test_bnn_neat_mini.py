import torch
import random
import pytest
import sys
from bnn import BayesianNN
sys.path.insert(0, '/scratch/cluster/dylantw/Risto/init/bnn-neat-python')
import bnn_neat
from bnn_neat.config import Config
from bnn_neat.genome import DefaultGenome
from bnn_neat.reproduction import DefaultReproduction
from bnn_neat.species import DefaultSpeciesSet
from bnn_neat.stagnation import DefaultStagnation
from bnn_neat.genes import DefaultConnectionGene
from bnn_neat.reporting import ReporterSet
from bnn_neat.innovation import innovation_number_generator, innovation_history, get_innovation_number



@pytest.fixture
def config():
    """
    Sets up the NEAT configuration for testing by directly initializing the Config class.
    """
    # Set up NEAT configurations
    print('hello')
    config_path = "/scratch/cluster/dylantw/Risto/init/mini-config-feedforward"  # Ensure this file exists in the correct directory
    config = bnn_neat.Config(
        bnn_neat.DefaultGenome, 
        bnn_neat.DefaultReproduction, 
        bnn_neat.DefaultSpeciesSet, 
        bnn_neat.DefaultStagnation, 
        config_path
    )
    
    # Debug: Print all sections and their attributes
    print("NEAT Config - pop_size:", config.pop_size)
    print("NEAT Config - fitness_criterion:", config.fitness_criterion)
    print("NEAT Config - fitness_threshold:", config.fitness_threshold)
    print("NEAT Config - reset_on_extinction:", config.reset_on_extinction)
    print("NEAT Config - no_fitness_termination:", config.no_fitness_termination)

    print("genome_config:", config.genome_config)
    print("species_set_config:", config.species_set_config)
    print("stagnation_config:", config.stagnation_config)
    print("reproduction_config:", config.reproduction_config)

    config.genome_config.num_inputs = 7
    config.genome_config.num_outputs = 4
    
    return config

@pytest.fixture
def mock_genome(config):
    """
    Returns a mock genome object with predefined connections and nodes for testing purposes.
    """
    try:
        print(f"Config loaded with species_fitness_func: {config.stagnation_config.species_fitness_func}")
    except AttributeError:
        print("Config does not have stagnation_config or species_fitness_func")
        raise
    genome_id = 0
    genome = DefaultGenome(genome_id)
    genome.configure_new(config.genome_config)

    # Add mock connections
    for i in range(-config.genome_config.num_inputs, 0):
        for j in range(config.genome_config.num_outputs):
            conn_key = (i, j)
            innovation_number = get_innovation_number(i, j)
            conn_gene = DefaultConnectionGene(conn_key, innovation_number)
            conn_gene.weight_mu = torch.randn(1).item()
            conn_gene.weight_sigma = torch.rand(1).item()
            conn_gene.enabled = True
            genome.connections[conn_key] = conn_gene

    return genome

@pytest.fixture
def bnn(mock_genome, config):
    """
    Initializes a BayesianNN instance using the mock genome and config.
    """

    config.genome_config.num_inputs = 7
    config.genome_config.num_outputs = 4
    return BayesianNN(mock_genome, config)

def test_evolutionary_architectures(bnn, config, mock_genome):
    """
    Test evolution over a few generations, printing each new architecture.
    """

    num_generations = 20
    reporters = ReporterSet()  # Empty list for reporters
    stagnation = DefaultStagnation(config.stagnation_config, reporters)
    species_set = DefaultSpeciesSet(config.species_set_config, reporters)
    reproduction = DefaultReproduction(config=config.reproduction_config, stagnation=stagnation, reporters=reporters)

    # Initialize the initial population with multiple mock genomes
    initial_population = {}
    num_initial_genomes = config.pop_size  # Initialize with pop_size genomes
    for gid in range(num_initial_genomes):
        genome = DefaultGenome(gid)
        genome.configure_new(config.genome_config)
        # Assign varying fitness levels to introduce diversity
        genome.fitness = random.uniform(0, 100)
        initial_population[gid] = genome
    
    population = initial_population

    # **Key Step: Assign Initial Population to Species Set**
    species_set.speciate(config, population, generation=0)
    print(f"Initial Species Count: {len(species_set.species)}")
    for sid, species in species_set.species.items():
        print(f"Species {sid}: {len(species.members)} members")

    for generation in range(num_generations):
        print(f"\n=== Generation {generation + 1} ===")

        try:
            # Call the reproduce method with all required arguments
            population = reproduction.reproduce(
                config=config,
                species=species_set,
                pop_size=config.pop_size,
                generation=generation + 1  # Assuming generations start at 1
            )
            print(f"Generation {generation + 1} completed successfully.")
            print(f"Population Size After Reproduction: {len(population)}")

            # Assert population size consistency
            assert len(population) == config.pop_size, f"Expected population size {config.pop_size}, got {len(population)}"

            # Assign fitness to new genomes
            for gid, genome in population.items():
                # Assign varying fitness levels to introduce diversity
                genome.fitness = random.uniform(0, 100)
                # Prune the genome
                pruned_genome = genome.get_pruned_copy(config.genome_config)
                # Assign the original key to the pruned genome
                pruned_genome.key = genome.key
                # Copy fitness and any other necessary attributes
                pruned_genome.fitness = genome.fitness
                pruned_genome.mutation_history = genome.mutation_history
                # Replace the genome in the population with the pruned version
                population[gid] = pruned_genome

            species_set.speciate(config, population, generation=generation + 1)


            # Log species details
            print(f"Number of Species After Reproduction: {len(species_set.species)}")
            for sid, species in species_set.species.items():
                print(f"Species {sid}: {len(species.members)} members")
        except TypeError as e:
            pytest.fail(f"Reproduce method failed: {e}")
        except ValueError as e:
            pytest.fail(f"Reproduce method encountered a ValueError: {e}")

        # Initialize and test each genome as a BNN architecture
        for genome_id, genome in population.items():
            print(f"\nTesting architecture for Genome {genome_id}:")
            #genome.prune_unused_genes(config.genome_config)
            bnn_instance = BayesianNN(genome, config)
            bnn_instance.build_network(config)
            bnn_instance.print_network_architecture()
            print(f"Genome {genome_id}: Mutation history")
            for mutation in genome.mutation_history:
                print(mutation)

            # Perform a simple forward pass with random data
            bnn_history = [{"response_embedding": [0.1],
                            "emotional_and_ethical_score": 0.5,
                            "environment_danger_score": 0.4,
                            "survived": 1,
                            "agent": "Strong"}] * 10
            output = bnn_instance.forward(bnn_history)
            print(f"Output shape: {output.shape}")

            # Ensure output is a torch.Tensor
            assert isinstance(output, torch.Tensor)

def test_svi_training(bnn):
    """
    Test the SVI optimization step.
    """
    bnn_history = [{"response_embedding": [0.1],
                    "emotional_and_ethical_score": 0.5,
                    "environment_danger_score": 0.4,
                    "survived": 1,
                    "agent": "Strong"}]

    loss = bnn.svi_step(bnn_history)
    print(f"SVI Loss: {loss}")
    assert isinstance(loss, float)
    assert loss >= 0.0

if __name__ == '__main__':
    pytest.main(["-s"])

