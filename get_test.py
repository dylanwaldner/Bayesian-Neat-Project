import configparser
import pandas as pd
import copy

# Read and parse the configuration file
config_file = 'config-feedforward'  # Replace with your actual config file name

config = configparser.ConfigParser()
config.optionxform = str  # Preserve the case of keys
config.read_string("""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1.0
pop_size              = 150
reset_on_extinction   = True

[DefaultGenome]
# Activation and aggregation functions
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
num_inputs              = 7686
num_outputs             = 4
initial_connection      = full

feed_forward            = True

enabled_default         = True
enabled_mutate_rate     = 0.1

node_add_prob           = 0.2
node_delete_prob        = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func    = max
max_stagnation          = 15
species_elitism         = 2

[DefaultReproduction]
elitism                 = 1
survival_threshold      = 0.2
""")

# Extract hyperparameters and default values
hyperparameters = {}
for section in config.sections():
    for key in config[section]:
        value = config[section][key]
        hyperparameters[f"{section}.{key}"] = value

# Define the hyperparameters you want to test and their test values
hyperparameter_tests = {
    'NEAT.pop_size': ['150', '50', '300'],  # Include default value '150'
    'DefaultSpeciesSet.compatibility_threshold': ['3.0', '1.5', '4.5'],  # Default is '3.0'
    'DefaultGenome.conn_add_prob': ['0.3', '0.1', '0.5'],  # Default is '0.3'
    'DefaultGenome.conn_delete_prob': ['0.2', '0.1', '0.3'],  # Default is '0.2'
    'DefaultGenome.node_add_prob': ['0.2', '0.05', '0.35'],  # Default is '0.2'
    'DefaultGenome.node_delete_prob': ['0.1', '0.05', '0.2'],  # Default is '0.1'
    # Add more hyperparameters and their test values as needed
}

# Remove any hyperparameters not in the config
hyperparameter_tests = {k: v for k, v in hyperparameter_tests.items() if k in hyperparameters}

default_settings = hyperparameters.copy()

# List to hold all test cases
test_cases = []

# Test ID counter
test_id = 1

for hyperparam, test_values in hyperparameter_tests.items():
    for value in test_values:
        if value == default_settings[hyperparam]:
            continue  # Skip the default value to ensure only one variation per test

        # Create a new test case with default settings
        test_case = default_settings.copy()

        # Vary only the current hyperparameter
        test_case[hyperparam] = value

        # Add test case metadata
        test_case['Test_ID'] = test_id
        test_case['Varied_Hyperparameter'] = hyperparam
        test_case['Varied_Value'] = value

        # Append to test cases
        test_cases.append(test_case)
        test_id += 1

# Convert list of dictionaries to a DataFrame
df = pd.DataFrame(test_cases)

# Reorder columns to have Test_ID and Varied_Hyperparameter at the front
columns = ['Test_ID', 'Varied_Hyperparameter', 'Varied_Value'] + [col for col in df.columns if col not in ['Test_ID', 'Varied_Hyperparameter', 'Varied_Value']]
df = df[columns]

# Export to Excel
output_excel_file = 'neat_hyperparameter_tests.xlsx'
df.to_excel(output_excel_file, index=False)
print(f"Excel file '{output_excel_file}' has been created with {len(test_cases)} test cases.")

