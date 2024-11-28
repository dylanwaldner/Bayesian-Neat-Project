import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.infer
from pyro.optim import Adam
from pyro.ops.indexing import Vindex

import numpy as np
import sys
import bnn_neat
from bnn.bnn_utils import get_activation_function, get_aggregation_function
from bnn.attention import compute_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Bayesian Neural Network that will be evolved by NEAT
class BayesianNN(nn.Module):
    def __init__(self, genome, config, attention_layers=None, lr=0.00001):
        super(BayesianNN, self).__init__()

        self.genome = genome
        self.config = config
        self.connections = {}  # To store connections with their properties
        self.nodes = {}        # To store node information if needed

        self.input_size = 7686

        self.dropout = nn.Dropout(p=0.1)

        self.query_proj = nn.Linear(self.input_size, self.input_size).to(device)
        self.key_proj = nn.Linear(self.input_size, self.input_size).to(device)
        self.value_proj = nn.Linear(self.input_size, self.input_size).to(device)

        self.learning_rate = lr

        self.optimizer = Adam({"lr": self.learning_rate})
        self.num_particles = 5 

        self.batch_indices = []

        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO(num_particles=self.num_particles, vectorize_particles=True))

        self.vectorize_particles = False
        self.model_output = None  # Add this line to initialize the attribute
        self.current_index = None
        self.bnn_history = None

        # Load the attention layers if provided
        if attention_layers:
            self.query_proj.load_state_dict(attention_layers['query_proj'])
            self.key_proj.load_state_dict(attention_layers['key_proj'])
            self.value_proj.load_state_dict(attention_layers['value_proj'])

        # Build the network based on the genome
        self.build_network(config)

        # Initialize node activation functions
        self.node_activation_funcs = {}

        self.input_matrix = None  # Matrix initialized later
        self.last_update_index = 0
        

    def print_network_architecture(self):
        # Limit floating-point precision for readability
        precision = 4
        architecture_details = []

        # Title
        architecture_details.append("=== Network Architecture ===\n")

        # Node Biases with limited precision
        architecture_details.append("Node Biases:")
        for node_id, bias_info in self.nodes.items():
            bias_mu = round(bias_info['bias_mu'], precision)
            bias_sigma = round(bias_info['bias_sigma'], precision)
            architecture_details.append(f"Node ID: {node_id}, Bias Mu: {bias_mu}, Bias Sigma: {bias_sigma}")

        # Connection Weights in table format with limited precision
        architecture_details.append("\nConnection Weights:")
        architecture_details.append(f"{'From Node':<10} {'To Node':<8} {'Weight Mu':<10} {'Weight Sigma':<12} {'Enabled':<8}")
        for conn_key, conn_data in self.connections.items():
            weight_mu = round(conn_data['weight_mu'], precision)
            weight_sigma = round(conn_data['weight_sigma'], precision)
            enabled = conn_data['enabled']
            architecture_details.append(f"{conn_key[0]:<10} {conn_key[1]:<8} {weight_mu:<10} {weight_sigma:<12} {enabled:<8}")

        # Network nodes and connections overview
        architecture_details.append("\nInput Nodes:")
        architecture_details.append(", ".join(map(str, self.input_nodes)))
        architecture_details.append("Output Nodes:")
        architecture_details.append(", ".join(map(str, self.output_nodes)))

        # Summarize all connections to make large networks readable
        architecture_details.append("\nConnections (summary):")
        total_connections = len(self.connections)
        if total_connections > 50:
            for i, (conn_key, conn_data) in enumerate(self.connections.items()):
                if i < 5 or i >= total_connections - 5:  # First and last few connections
                    weight_mu = round(conn_data['weight_mu'], precision)
                    weight_sigma = round(conn_data['weight_sigma'], precision)
                    enabled = conn_data['enabled']
                    architecture_details.append(
                        f"From Node {conn_key[0]} to Node {conn_key[1]} - Weight Mu: {weight_mu}, Weight Sigma: {weight_sigma}, Enabled: {enabled}"
                    )
                elif i == 5:
                    architecture_details.append("...")
        else:
            for conn_key, conn_data in self.connections.items():
                weight_mu = round(conn_data['weight_mu'], precision)
                weight_sigma = round(conn_data['weight_sigma'], precision)
                enabled = conn_data['enabled']
                architecture_details.append(
                    f"From Node {conn_key[0]} to Node {conn_key[1]} - Weight Mu: {weight_mu}, Weight Sigma: {weight_sigma}, Enabled: {enabled}"
                )
                architecture_details.append(f"New debugging: {conn_key}: {conn_data}")

        # Attention Layer Details
        architecture_details.append("\nAttention Layer Details:")
        architecture_details.append(f"Query Projection: {self.query_proj}")
        architecture_details.append(f"Key Projection: {self.key_proj}")
        architecture_details.append(f"Value Projection: {self.value_proj}")

        # SVI Parameters
        architecture_details.append("\nSVI Parameters:")
        architecture_details.append(f"Optimizer: {self.optimizer}")
        architecture_details.append(f"Loss Function: {type(self.svi.loss)}")

        # Posterior Distributions
        architecture_details.append("\nPosterior Distributions:")
        for node_id, bias_info in self.nodes.items():
            architecture_details.append(f"Node {node_id} Bias Mu: {bias_info['bias_mu']}, Bias Sigma: {bias_info['bias_sigma']}")
        for conn_key, conn_data in self.connections.items():
            architecture_details.append(f"Connection {conn_key}: Weight Mu: {conn_data['weight_mu']}, Weight Sigma: {conn_data['weight_sigma']}")


        # End title
        architecture_details.append("\n=== End of Network Architecture ===")

        # Join all details into a single string
        return "\n".join(architecture_details)


    def build_network(self, config):
        """
        Builds the neural network layers based on the genome's nodes and connections.
        """
        # Extract the number of input and output nodes from the genome
        num_inputs = self.config.genome_config.num_inputs
        #print(num_inputs)
        num_outputs = self.config.genome_config.num_outputs
        #print(num_outputs)
        # Generate node IDs
        input_node_ids = list(range(-num_inputs, 0))
        output_node_ids = list(range(num_outputs))

        # Store input and output nodes directly as attributes for easy access
        self.input_nodes = input_node_ids
        self.output_nodes = output_node_ids

        # Hidden nodes from the genome
        hidden_node_ids = [nid for nid in self.genome.nodes.keys() if nid >= 0 and nid not in output_node_ids]

        #print("NUMBER OF HIDDEN NODES: ", len(hidden_node_ids))

        # Combine all node IDs
        all_node_ids = input_node_ids + hidden_node_ids + output_node_ids

        # Build the node_id_to_index mapping
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
        self.node_id_to_index = node_id_to_index  # Store the mapping

        # Set input and output indices
        self.input_indices = config.genome_config.input_keys  # This should be [-1, -2, -3, -4, -5, -6, -7]
        self.output_indices = config.genome_config.output_keys

        # Identify active connections (enabled)
        active_connections = {k: v for k, v in self.genome.connections.items() if v.enabled}

        # Store connections with their properties
        self.connections = {}
        for conn_key, conn in active_connections.items():
            self.connections[conn_key] = {
                'weight_mu': getattr(conn, 'weight_mu', 0.0),
                'weight_sigma': getattr(conn, 'weight_sigma', 1.0),
                'enabled': getattr(conn, 'enabled', True)
            }

        # For each node, gather bias info
        self.nodes = {}
        for node_id in all_node_ids:
            if node_id < 0:
                continue  # Input nodes may not have biases

            node_gene = self.genome.nodes[node_id]

            # Map activation and aggregation function names to actual functions
            activation_func = get_activation_function(node_gene.activation)
            aggregation_func = get_aggregation_function(node_gene.aggregation)

            self.nodes[node_id] = {
                'bias_mu': getattr(self.genome.nodes[node_id], 'bias_mu', 0.0),
                'bias_sigma': getattr(self.genome.nodes[node_id], 'bias_sigma', 1.0),
                'activation_func': activation_func,
                'aggregation_func': aggregation_func
            }

        # Define the weight matrices
        num_nodes = len(all_node_ids)
        self.weight_mu_matrix = torch.zeros((num_nodes, num_nodes))
        self.weight_sigma_matrix = torch.ones((num_nodes, num_nodes)) * 0.1  # Default sigma

        for conn_key, conn_data in self.connections.items():
            in_idx = node_id_to_index[conn_key[0]]
            out_idx = node_id_to_index[conn_key[1]]
            # Convert conn_data['weight_mu'] and conn_data['weight_sigma'] to tensors before assignment
            self.weight_mu_matrix[in_idx, out_idx] = torch.tensor(conn_data['weight_mu'], dtype=torch.float32)
            self.weight_sigma_matrix[in_idx, out_idx] = torch.tensor(conn_data['weight_sigma'], dtype=torch.float32)

        # Optionally, print the network architecture for debugging
        # self.print_network_architecture()

    def get_connections(self):
        """
        Returns the connections with their properties.
        """
        return self.connections

    def update_matrix(self, bnn_history, current_index=None, device=None):
        if current_index is None:
            current_index = len(bnn_history) - 1

        if device is None:
            device = torch.device("cuda")

        if isinstance(bnn_history[current_index]["response_embedding"], torch.Tensor):
            current_embedding = bnn_history[current_index]["response_embedding"].clone().detach().to(device)
        else:
            current_embedding = torch.tensor(bnn_history[current_index]["response_embedding"], device=device)


        agent_mapping = {
            "Storyteller": torch.tensor([1, 0], device=device),
            "Strong": torch.tensor([0, 1], device=device),
        }

        rows = []

        # Input matrix construction
        for i in range(self.last_update_index, current_index + 1):
            dictionary = bnn_history[i]

            row = torch.cat([
                agent_mapping[dictionary["agent"]].clone().detach().float(),
                torch.tensor(dictionary["response_embedding"], device=device),
                torch.tensor([dictionary["emotional_and_ethical_score"]], device=device),
                torch.tensor([dictionary["environment_danger_score"]], device=device),
                torch.tensor([dictionary["survived"]], device=device),
                torch.tensor([0.0], device=device)  # Placeholder for relevance score
            ])

            rows.append(row)

        # Convert list of tensors to a 2D tensor
        new_rows = torch.stack(rows)

        # Update self.input_matrix
        if self.input_matrix is None:
            self.input_matrix = new_rows  # First time, set input_matrix to new_rows
        else:
            self.input_matrix = torch.cat([self.input_matrix, new_rows], dim=0)  # Append new_rows

        # Recalculate relevance scores for previously stored rows
        for i in range(len(self.input_matrix)):
            past_embedding = self.input_matrix[i][2:-4]  # Adjust slice indices as needed
            relevance_score = torch.norm(current_embedding - past_embedding)
            relevance_score = 1 / (1 + relevance_score)  # Normalize relevance
            self.input_matrix[i][-1] = relevance_score  # Update the relevance score

        self.last_update_index = current_index + 1
        self.input_matrix = self.input_matrix.clone().detach().float().to(device)

        # After building the input_matrix, we now know the input size
        input_size = self.input_matrix.shape[1]
        num_memories = self.input_matrix.shape[0]

    def update_neat_config(config, num_inputs, num_outputs):
        """
        Updates the NEAT configuration to match the BNN's input and output sizes.

        Args:
            config: The NEAT configuration object.
            num_inputs (int): Number of input nodes in the BNN.
            num_outputs (int): Number of output nodes in the BNN.
        """
        # Update the number of inputs and outputs
        config.genome_config.num_inputs = num_inputs
        config.genome_config.num_outputs = num_outputs

        # Optionally, you can adjust initial connection settings or other parameters
        # For example, you might want to set initial_connection to 'full' or 'partial'
        # depending on your preference
        config.genome_config.initial_connection = 'full'  # Or 'partial' or as needed

        # If your BNN uses specific activation functions, you can set them here
        # config.genome_config.activation_default = 'sigmoid'  # Or other activation functions

    def topological_sort(self):
        nodes = list(self.nodes.keys())
        connections = [
            (in_node, out_node)
            for (in_node, out_node), conn in self.connections.items()
            if conn['enabled']
        ]

        # Build the dependency graph
        dependency_graph = {node_id: [] for node_id in nodes}
        for (in_node, out_node) in connections:
            dependency_graph[out_node].append(in_node)

        visited = set()
        order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for dep_node in dependency_graph.get(node, []):
                dfs(dep_node)
            order.insert(0, node)  # Prepend node to order

        for node in nodes:
            if node not in visited:
                dfs(node)

        order.reverse()
        return order

    def forward_svi(self, bnn_history, weights_samples, bias_samples, device=None):
        torch.autograd.set_detect_anomaly(True)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_indices = self.batch_indices

        num_particles = next(iter(weights_samples.values())).shape[0]
        batch_size = len(batch_indices)
        input_size = self.input_matrix.size(1)
        

        # Build id_to_last_index mapping
        id_to_last_index = {}
        for idx, entry in enumerate(self.bnn_history):
            id = entry['id']
            id_to_last_index[id] = idx

        # Initialize combined_context tensor
        combined_context_list = []

        # Loop over batch entries
        for idx, batch_idx in enumerate(batch_indices):
            # Get the last index in bnn_history for the current id
            current_index = id_to_last_index[batch_idx]  # Correctly maps to the last index for this id

            # Prepare input matrix and mask for the current entry
            relevant_input = self.input_matrix[:current_index + 1, :].clone().detach().float().to(device)
            mask = (relevant_input != -1).float().to(device)
            masked_input = relevant_input * mask  # Shape: (sequence_length, input_size)

            # Create storyteller mask
            storyteller_mask = torch.tensor(
                [entry['agent'] == "Storyteller" for entry in self.bnn_history[:current_index + 1]],
                dtype=torch.bool,
                device=device
            )

            # Compute combined context vector
            combined_context = compute_attention(
                masked_input,
                self.query_proj,
                self.key_proj,
                self.value_proj,
                self.dropout,
                storyteller_mask,
                scaling_factor=0.3,
                device=device
            )  # Shape: (input_size,)

            combined_context_list.append(combined_context.squeeze())  # Shape: (input_size,)

        # Stack combined_contexts into a tensor
        combined_context = torch.stack(combined_context_list, dim=0)  # Shape: (batch_size, input_size)
        print(f"Combined context shape after stacking: {combined_context.shape}")

        # Expand combined_context to match num_particles
        combined_context = combined_context.unsqueeze(0).expand(num_particles, -1, -1)  # Shape: [num_particles, batch_size, input_size]
        print(f"Combined context shape after expand: {combined_context.shape}")
        # Initialize node activations using context vector for each input node
        node_activations = {}
        for idx, node_id in enumerate(self.input_indices):
            input_value = combined_context[:, :, idx]  # Shape: [num_particles, batch_size]
            node_activations[node_id] = input_value

        # Process nodes in topological order
        node_order = self.topological_sort()
        for node_id in node_order:
            if node_id in self.input_indices:
                continue  # Skip input nodes

            incoming_values = []
            for conn_key, conn_data in self.connections.items():
                in_node, out_node = conn_key
                if out_node == node_id and conn_data['enabled']:
                    weight = weights_samples[conn_key][:, None]  # Shape: [num_particles, 1]
                    input_value = node_activations[in_node]  # Shape: [num_particles, batch_size]

                    # Expand weight to match batch_size
                    incoming_value = weight * input_value  # Shape: [num_particles, batch_size]
                    incoming_values.append(incoming_value)

            # Apply the node's aggregation function
            if incoming_values:
                stacked_values = torch.stack(incoming_values, dim=-1)  # Shape: [num_particles, batch_size, num_incoming]
                aggregation_func = self.nodes[node_id].get('aggregation_func', torch.sum)
                # Handle functions that return (values, indices), like max and min
                if aggregation_func in [torch.max, torch.min]:
                    total_input, _ = aggregation_func(stacked_values, dim=-1)  # Shape: [num_particles, batch_size]
                elif aggregation_func == torch.prod:
                    total_input = aggregation_func(stacked_values, dim=-1)  # Shape: [num_particles, batch_size]
                else:
                    total_input = aggregation_func(stacked_values, dim=-1)  # Shape: [num_particles, batch_size]
            else:
                total_input = torch.zeros(num_particles, batch_size, device=device)  # Shape: [num_particles, batch_size]

            # Add bias
            bias = bias_samples.get(node_id, torch.zeros(num_particles, device=device))[:, None]  # Shape: [num_particles, 1]

            total_input = total_input + bias  # Shape: [num_particles, batch_size]

            # Apply activation function
            if node_id in self.output_nodes:
                activation = total_input  # Shape: [num_particles, batch_size]
            else:
                activation_func = self.nodes[node_id].get('activation_func', torch.relu)
                activation = activation_func(total_input)  # Shape: [num_particles, batch_size]

            node_activations[node_id] = activation  # Shape: [num_particles, batch_size]

        # Collect outputs
        outputs = torch.stack([node_activations[node_id] for node_id in self.output_nodes], dim=-1)  # Shape: [num_particles, batch_size, num_outputs]
        print("OUTPUTS SHAPE: ", outputs.shape)

        outputs = outputs.to(device)  # Ensure outputs are on the correct device
        outputs = outputs.float()
        return outputs



    def forward_bce(self, bnn_history, current_index=None, num_samples=1, device=None):
        torch.autograd.set_detect_anomaly(True)

        # Retrieve the device from the model's parameters
        if device is None:
            device = torch.device("cuda")  # Default to cuda if device is not provided

        if current_index is None:
            if self.current_index is not None:
                current_index = self.current_index
            if self.current_index is None:
                current_index = len(bnn_history) - 1
        else:
            self.bnn_history = bnn_history

        # Ensure self.input_matrix contains the full history only once
        if len(bnn_history) > self.last_update_index:
            self.update_matrix(self.bnn_history, current_index, device)

        # Prepare input matrix and mask
        self.input_matrix = self.input_matrix.clone().detach().float().to(device)
        mask = (self.input_matrix != -1).float().to(device)
        masked_input = self.input_matrix * mask  # Shape: (full_sequence_length, input_size)

        # Slice input based on current_index if provided
        relevant_input = masked_input[:current_index + 1, :]  # Shape: (sequence_length, input_size)
        mask = mask[:current_index + 1, :]

        sequence_length, input_size = relevant_input.shape

        # Create storyteller mask
        storyteller_mask = torch.tensor(
            [entry['agent'] == "Storyteller" for entry in self.bnn_history[:current_index + 1]],
            dtype=torch.bool,
            device=device
        )

        # Compute combined context vector
        combined_context = compute_attention(
            relevant_input,
            self.query_proj,
            self.key_proj,
            self.value_proj,
            self.dropout,
            storyteller_mask,
            scaling_factor=0.3,
            device=device
        )

        # Expand for Monte Carlo samples
        context_vector = combined_context.expand(num_samples, -1)

        # Initialize node activations using context vector for each input node
        node_activations = {}
        for idx, node_id in enumerate(self.input_indices):
            input_value = context_vector[:, idx]  # Shape: (num_samples,)
            node_activations[node_id] = input_value

        # Sample weights
        sampled_weights = {}
        for conn_key, conn_data in self.connections.items():
            weight_mu = torch.tensor(conn_data['weight_mu'], dtype=torch.float32, device=device)
            weight_sigma = torch.tensor(conn_data['weight_sigma'], dtype=torch.float32, device=device)
            weight_sigma = torch.clamp(weight_sigma, min=1e-5)
            weight_dist = dist.Normal(weight_mu, weight_sigma)
            weight_samples = pyro.sample(f"w_{conn_key}", weight_dist.expand([num_samples]))
            sampled_weights[conn_key] = weight_samples  # Shape: (num_samples,)

        # Sample biases
        sampled_biases = {}
        for node_id, node_data in self.nodes.items():
            bias_mu = torch.tensor(node_data['bias_mu'], dtype=torch.float32, device=device)
            bias_sigma = torch.tensor(node_data['bias_sigma'], dtype=torch.float32, device=device)
            bias_sigma = torch.clamp(bias_sigma, min=1e-5)
            bias_dist = dist.Normal(bias_mu, bias_sigma)
            bias_samples = pyro.sample(f"b_{node_id}", bias_dist.expand([num_samples]))
            sampled_biases[node_id] = bias_samples  # Shape: (num_samples,)

        # Process nodes in topological order
        node_order = self.topological_sort()
        for node_id in node_order:
            if node_id in self.input_indices:
                continue  # Skip input nodes

            incoming_values = []
            for conn_key, conn_data in self.connections.items():
                in_node, out_node = conn_key
                if out_node == node_id and conn_data['enabled']:
                    weight = sampled_weights[conn_key]  # Shape: (num_samples,)
                    if in_node not in node_activations:
                        print(f"Activation for node {in_node} not found in node_activations.")
                        print(f"Current node_id: {node_id}")
                        print(f"Available node_activations keys: {list(node_activations.keys())}")
                        raise KeyError(f"Activation for node {in_node} is not available.")
                    input_value = node_activations[in_node]  # Shape: (num_samples,)
                    incoming_values.append(weight * input_value)  # Shape: (num_samples,)

            # Apply the node's aggregation function
            if incoming_values:
                stacked_values = torch.stack(incoming_values, dim=0)  # Shape: (num_incoming, num_samples)
                aggregation_func = self.nodes[node_id].get('aggregation_func', torch.sum)
                # Handle functions that return (values, indices), like max and min
                if aggregation_func in [torch.max, torch.min]:
                    total_input, _ = aggregation_func(stacked_values, dim=0)  # Shape: (num_samples,)

                elif aggregation_func == torch.prod:
                    # Clone the tensor before using torch.prod to avoid in-place modification errors
                    total_input = aggregation_func(stacked_values.clone(), dim=0)  # Shape: (num_samples,)

                else:
                    total_input = aggregation_func(stacked_values, dim=0)  # Shape: (num_samples,)
            else:
                total_input = torch.zeros(num_samples, device=device)  # Shape: (num_samples,)

            # Add bias
            bias = sampled_biases.get(node_id, torch.zeros(num_samples, device=device))
            total_input = total_input + bias  # Shape: (num_samples,)

            # Apply activation function
            if node_id in self.output_nodes:
                activation = total_input
            else:
                activation_func = self.nodes[node_id].get('activation_func', torch.relu)
                activation = activation_func(total_input)


            # Check for NaNs
            if torch.isnan(activation).any():
                print(f"NaN detected in activation of node {node_id}")
                raise ValueError("NaN detected in activation.")

            node_activations[node_id] = activation  # Shape: (num_samples,)

        # Collect outputs
        outputs = torch.stack([node_activations[node_id] for node_id in self.output_nodes], dim=1)
        outputs = outputs.to(device)  # Ensure outputs are on the correct device

        expected_shape = (num_samples, len(self.output_nodes))
        assert outputs.shape == expected_shape, f"Unexpected output shape: {outputs.shape}"
        outputs = outputs.float()
        print("MODEL OUTPUTS: ", outputs)
        probabilities = torch.sigmoid(outputs)
        print("Probabilities: ", probabilities)
        predictions = (probabilities > 0.5).float()
        print("Predictions: ", predictions)
        return outputs

    def prepare_weights(self):
        printed = False
        self.prepared_weights = {}
        for conn_key, conn_data in self.connections.items():
            weight_mu = conn_data["weight_mu"]
            weight_sigma = conn_data["weight_sigma"]
            # Ensure weight_mu and weight_sigma are scalars
            if isinstance(weight_mu, (list, np.ndarray)) and len(weight_mu) == 1:
                print("Readjusting weight mu model")
                weight_mu = weight_mu[0]
            if isinstance(weight_sigma, (list, np.ndarray)) and len(weight_sigma) == 1:
                print("Readjusting weight sigma model")
                weight_sigma = weight_sigma[0]

            # Convert to tensors
            weight_mu_tensor = torch.tensor(weight_mu, dtype=torch.float32, device=device)
            weight_sigma_tensor = torch.tensor(weight_sigma, dtype=torch.float32, device=device).clamp(min=1e-5)

            # Assert correct shape
            assert weight_mu_tensor.dim() == 0, f"Unexpected shape for weight_mu: {weight_mu_tensor.shape}, expected scalar"
            assert weight_sigma_tensor.dim() == 0, f"Unexpected shape for weight_sigma: {weight_sigma_tensor.shape}, expected scalar"

            # Store prepared weights
            self.prepared_weights[conn_key] = {
                "mu": weight_mu_tensor,
                "sigma": weight_sigma_tensor,
            }

    def model(self, x_data, y_data):
        # Prepare weights if not already done
        if not hasattr(self, "prepared_weights"):
            self.prepare_weights()

        x_data = x_data.to(device, dtype=torch.float32)  # Shape: [batch_size, input_size]
        y_data = y_data.to(device, dtype=torch.float32)  # Shape: [batch_size, num_outputs]
        if y_data.dim() == 1:
            y_data = y_data.unsqueeze(1)  # Adjust y_data shape to [batch_size, num_outputs]

        batch_size = y_data.size(0)
        num_particles = self.num_particles

        # Sample weights and biases inside the particles plate
        with pyro.plate("particles", num_particles, dim=-3):
            expected_sample_shape = torch.Size([num_particles])
            # Sample weights
            printed = False
            sampled_weights = {}
            for conn_key, prepared_data in self.prepared_weights.items():
                weight_mu = prepared_data["mu"]
                weight_sigma = prepared_data["sigma"]
                weight_samples = pyro.sample(f"w_{conn_key}", dist.Normal(weight_mu, weight_sigma))
                if not printed:
                    print("Weights sample in model: ", weight_samples.shape) 
                    printed = True

                if weight_samples.shape != expected_sample_shape:
                    # Squeeze out singleton dimensions
                    weight_samples = weight_samples.squeeze()

                    # Handle specific cases for shape corrections
                    if weight_samples.shape == (num_particles, num_particles):
                        weight_samples = weight_samples[..., 0]  # Select the first "list" of samples
                    elif weight_samples.shape == (num_particles, num_particles, num_particles):
                        weight_samples = weight_samples[..., 0, 0]  # Select the first "list" of samples

                assert weight_samples.shape == expected_sample_shape, \
                    print(f"Unexpected shape for weight {conn_key}: {weight_samples.shape}, expected: {expected_sample_shape}")

                sampled_weights[conn_key] = weight_samples  # Shape: [num_particles]

            # Sample biases
            sampled_biases = {}
            for node_id, node_data in self.nodes.items():
                bias_mu = torch.tensor(node_data["bias_mu"], dtype=torch.float32, device=device)
                bias_sigma = torch.tensor(node_data["bias_sigma"], dtype=torch.float32, device=device).clamp(min=1e-5)
                bias_samples = pyro.sample(f"b_{node_id}", dist.Normal(bias_mu, bias_sigma))

                if bias_samples.shape != expected_sample_shape:
                    # Squeeze out singleton dimensions
                    bias_samples = bias_samples.squeeze()

                    # Handle specific cases for shape corrections
                    if bias_samples.shape == (num_particles, num_particles):
                        bias_samples = bias_samples[..., 0]  # Select the first "list" of samples
                    elif bias_samples.shape == (num_particles, num_particles, num_particles):
                        bias_samples = bias_samples[..., 0, 0]  # Select the first "list" of samples

                assert bias_samples.shape == expected_sample_shape, \
                    print(f"Unexpected shape for bias {node_id}: {bias_samples.shape}, expected: {expected_sample_shape}")

                sampled_biases[node_id] = bias_samples  # Shape: [num_particles]

            # Now process the data inside a data plate
            with pyro.plate("data", batch_size, dim=-2):
                # Proceed with forward pass
                logits = self.forward_svi(
                    self.bnn_history,
                    weights_samples=sampled_weights,
                    bias_samples=sampled_biases
                ).to(device)  # Shape: [num_particles, batch_size, num_outputs]

                self.model_output = logits.mean(0) # Not sure how to handle this yet,shape = [batch_size, 4]?

                # Expand y_data to match logits shape
                y_data_expanded = y_data.unsqueeze(0).expand(num_particles, -1, -1)  # Shape: [num_particles, batch_size, num_outputs]
                assert list(y_data_expanded.shape) == list(logits.shape), \
                    print(f"Unexpected shape for expanded y_data: {y_data_expanded.shape}, expected: {logits.shape}")

                # Sample the observations
                pyro.sample("obs", dist.Bernoulli(logits=logits).to_event(2), obs=y_data_expanded)


    def guide(self, x_data, y_data):
        num_particles = self.num_particles

        # Register parameters
        for conn_key in self.connections:
            weight_mu_value = float(self.connections[conn_key]['weight_mu'])
            weight_sigma_value = float(self.connections[conn_key]['weight_sigma'])

            # Assertions to ensure the parameters are scalars
            assert isinstance(weight_mu_value, float), f"weight_mu for {conn_key} is not a scalar. Got: {type(weight_mu_value)}"
            assert isinstance(weight_sigma_value, float), f"weight_sigma for {conn_key} is not a scalar. Got: {type(weight_sigma_value)}"


            weight_mu_param = pyro.param(
                f"w_mu_{conn_key}",
                torch.tensor(weight_mu_value, device=device, dtype=torch.float32)
            )
            weight_sigma_param = pyro.param(
                f"w_sigma_{conn_key}",
                torch.tensor(weight_sigma_value, device=device, dtype=torch.float32),
                constraint=dist.constraints.positive
            )

            # Assertions to ensure the Pyro parameters have the correct shape and type
            assert weight_mu_param.shape == torch.Size([]), \
                f"weight_mu_param for {conn_key} has unexpected shape {weight_mu_param.shape}, expected scalar []"
            assert weight_sigma_param.shape == torch.Size([]), \
                f"weight_sigma_param for {conn_key} has unexpected shape {weight_sigma_param.shape}, expected scalar []"

        for node_id in self.nodes:
            bias_mu_value = float(self.nodes[node_id]['bias_mu'])
            bias_sigma_value = float(self.nodes[node_id]['bias_sigma'])

            bias_mu_param = pyro.param(
                f"b_mu_{node_id}",
                torch.tensor(bias_mu_value, device=device, dtype=torch.float32)
            )
            bias_sigma_param = pyro.param(
                f"b_sigma_{node_id}",
                torch.tensor(bias_sigma_value, device=device, dtype=torch.float32),
                constraint=dist.constraints.positive
            )

             # Assertions to ensure the Pyro parameters have the correct shape and type
            assert bias_mu_param.shape == torch.Size([]), \
                f"bias_mu_param for {node_id} has unexpected shape {bias_mu_param.shape}, expected scalar []"
            assert bias_sigma_param.shape == torch.Size([]), \
                f"bias_sigma_param for {node_id} has unexpected shape {bias_sigma_param.shape}, expected scalar []"


        # Sample weights and biases inside the particles plate
        with pyro.plate("particles", num_particles, dim=-3):
            expected_sample_shape = torch.Size([num_particles])
            # Sample weights
            printed = False
            for conn_key in self.connections:
                weight_mu = pyro.param(f"w_mu_{conn_key}")
                weight_sigma = pyro.param(f"w_sigma_{conn_key}")
                weight_samples = pyro.sample(f"w_{conn_key}", dist.Normal(weight_mu, weight_sigma))

                if weight_samples.shape != expected_sample_shape:
                    # Squeeze out singleton dimensions
                    weight_samples = weight_samples.squeeze()

                    # Handle specific cases for shape corrections
                    if weight_samples.shape == (num_particles, num_particles):
                        weight_samples = weight_samples[..., 0]  # Select the first "list" of samples
                    elif weight_samples.shape == (num_particles, num_particles, num_particles):
                        weight_samples = weight_samples[..., 0, 0]  # Select the first "list" of samples

                if not printed:
                    print("Weights sample in model: ", weight_samples.shape)
                    printed = True

                assert weight_samples.shape == expected_sample_shape, \
                    print(f"Unexpected shape for weight {conn_key}: {weight_samples.shape}, expected: {expected_sample_shape}")

            # Sample biases
            for node_id in self.nodes:
                bias_mu = pyro.param(f"b_mu_{node_id}")
                bias_sigma = pyro.param(f"b_sigma_{node_id}")
                bias_samples = pyro.sample(f"b_{node_id}", dist.Normal(bias_mu, bias_sigma))

                if bias_samples.shape != expected_sample_shape:
                    # Squeeze out singleton dimensions
                    bias_samples = bias_samples.squeeze()

                    # Handle specific cases for shape corrections
                    if bias_samples.shape == (num_particles, num_particles):
                        bias_samples = bias_samples[..., 0]  # Select the first "list" of samples
                    elif bias_samples.shape == (num_particles, num_particles, num_particles):
                        bias_samples = bias_samples[..., 0, 0]  # Select the first "list" of samples

                assert bias_samples.shape == expected_sample_shape, \
                    print(f"Unexpected shape for bias {node_id}: {bias_samples.shape}, expected: {expected_sample_shape}")



    def svi_step(self, bnn_history, ground_truth):
        if len(bnn_history) > self.last_update_index:
            self.update_matrix(bnn_history)

        self.current_index = len(bnn_history) - 1
        self.bnn_history = bnn_history
        self.vectorize_particles = True  # Enable vectorized particles

        # Determine the current batch size
        batch_size = len(ground_truth)

        # x_data is a dummy variable; can be None or adjusted if necessary
        x_data = torch.empty((batch_size, self.input_size), device=device).fill_(-1)
        y_data = torch.tensor(ground_truth, dtype=torch.float32).to(device)

        # Perform one step of SVI training
        try:
            loss = self.svi.step(x_data, y_data)
            print("LOSS: ", loss)
        except ValueError as e:
            print("ValueError in svi_step:", e)
            raise

        self.current_index = None
        self.vectorize_particles = False  # Reset after SVI step

        return loss

    def get_optimized_parameters(self):
        optimized_params = {}
        param_store = pyro.get_param_store()
        for name in param_store.get_all_param_names():
            param_value = param_store.get_param(name).detach().cpu().clone()
            optimized_params[name] = param_value
        return optimized_params

    def update_genome_parameters(self, genome, optimized_params):
        # Update genome's connection parameters
        for conn_key, conn in genome.connections.items():
            weight_mu_name = f"w_mu_{conn_key}"
            weight_sigma_name = f"w_sigma_{conn_key}"
            if weight_mu_name in optimized_params and weight_sigma_name in optimized_params:
                conn.weight_mu = optimized_params[weight_mu_name].item()
                conn.weight_sigma = optimized_params[weight_sigma_name].item()
        # Update genome's node parameters
        for node_id, node in genome.nodes.items():
            bias_mu_name = f"b_mu_{node_id}"
            bias_sigma_name = f"b_sigma_{node_id}"
            if bias_mu_name in optimized_params and bias_sigma_name in optimized_params:
                node.bias_mu = optimized_params[bias_mu_name].item()
                node.bias_sigma = optimized_params[bias_sigma_name].item()


    def sample_weights(self):
        # For each connection, sample weight from Normal distribution
        sampled_weights = {}
        for conn_key, conn_data in self.connections.items():
            weight_mu = conn_data['weight_mu']
            weight_sigma = conn_data['weight_sigma']
            weight_dist = dist.Normal(weight_mu, weight_sigma)
            weight_sample = pyro.sample(f"w_{conn_key}", weight_dist).to(device)
            sampled_weights[conn_key] = weight_sample
        return sampled_weights

    def compute_bce_loss(self, bnn_history, ground_truth_labels, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the model is in evaluation mode
        self.eval()

        # Update bnn_history as a class attribute
        if len(bnn_history) > self.last_update_index:
            self.update_matrix(bnn_history)

        self.current_index = len(bnn_history) - 1
        self.bnn_history = bnn_history

        # Create dummy x_data and move ground_truth_labels to the correct device
        x_data = torch.empty((1, self.input_size), device=device).fill_(-1)  # Dummy input tensor

        # Ensure ground_truth_labels is a tensor
        if isinstance(ground_truth_labels, list):
            ground_truth_labels = torch.tensor(ground_truth_labels, dtype=torch.float32)

        y_data = ground_truth_labels.to(device)

        # Print for debugging purposes (optional)
        print("x_data shape:", x_data.shape)
        print("y_data shape:", y_data.shape)

        # Perform forward pass to compute logits
        logits = self.forward_bce(x_data, device=device)

        logits = logits.squeeze()

        # Apply sigmoid activation to get probabilities
        probabilities = torch.sigmoid(logits)

        # Compute BCE loss
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(probabilities, y_data)

        # Reset the current index after computation
        self.current_index = None

        return loss.item(), probabilities
