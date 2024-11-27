def get_initial_rates():
    """
    Returns a dictionary containing the initial rates for parameters 
    with high starting values.
    """
    return {
        # Bias mutation and replacement
        "bias_mu_mutate_rate": 0.9,
        "bias_mu_mutate_power": 1.0,
        "bias_mu_replace_rate": 0.5,
        "bias_sigma_mutate_rate": 0.9,
        "bias_sigma_mutate_power": 0.9,
        "bias_sigma_replace_rate": 0.5,

        # Weight mutation and replacement
        "weight_mu_mutate_rate": 0.9,
        "weight_mu_mutate_power": 1.0,
        "weight_mu_replace_rate": 0.5,
        "weight_sigma_mutate_rate": 0.9,
        "weight_sigma_mutate_power": 1.0,
        "weight_sigma_replace_rate": 0.5,

        # Response mutation and replacement
        "response_mu_mutate_rate": 0.9,
        "response_mu_mutate_power": 0.9,
        "response_mu_replace_rate": 0.5,
        "response_sigma_mutate_rate": 0.9,
        "response_sigma_mutate_power": 0.5,
        "response_sigma_replace_rate": 0.5,

        # Node mutation
        "node_add_prob": 0.9,
        "node_delete_prob": 0.7,

        # Connection mutation
        "conn_add_prob": 0.9,
        "conn_delete_prob": 0.9,
    }


def get_final_rates():
    """
    Returns a dictionary containing the final rates for parameters 
    with low ending values for gradual decay.
    """
    return {
        # Bias mutation and replacement
        "bias_mu_mutate_rate": 0.1,
        "bias_mu_mutate_power": 0.1,
        "bias_mu_replace_rate": 0.1,
        "bias_sigma_mutate_rate": 0.1,
        "bias_sigma_mutate_power": 0.1,
        "bias_sigma_replace_rate": 0.1,

        # Weight mutation and replacement
        "weight_mu_mutate_rate": 0.1,
        "weight_mu_mutate_power": 0.1,
        "weight_mu_replace_rate": 0.1,
        "weight_sigma_mutate_rate": 0.1,
        "weight_sigma_mutate_power": 0.1,
        "weight_sigma_replace_rate": 0.1,

        # Response mutation and replacement
        "response_mu_mutate_rate": 0.1,
        "response_mu_mutate_power": 0.1,
        "response_mu_replace_rate": 0.1,
        "response_sigma_mutate_rate": 0.1,
        "response_sigma_mutate_power": 0.1,
        "response_sigma_replace_rate": 0.1,

        # Node mutation
        "node_add_prob": 0.1,
        "node_delete_prob": 0.1,

        # Connection mutation
        "conn_add_prob": 0.1,
        "conn_delete_prob": 0.1,
    }
