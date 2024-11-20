import sys
sys.path.insert(0, '/scratch/cluster/dylantw/Risto/init/bnn-neat-python')
import bnn_neat

# Define the function

def test_minimal_config_loading():
    config_path = "config-feedforward"
    config = bnn_neat.Config(bnn_neat.DefaultGenome, bnn_neat.DefaultReproduction, bnn_neat.DefaultSpeciesSet, bnn_neat.DefaultStagnation, config_path)
    assert hasattr(config, 'species_fitness_func')
    assert config.species_fitness_func == "max"

# Call the function
test_minimal_config_loading()

