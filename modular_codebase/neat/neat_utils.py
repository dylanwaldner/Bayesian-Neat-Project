def adjust_rates_proportional(config, neat_iteration, total_iterations, initial_rates, final_rates):
    """
    Adjust rates proportionally based on the initial values, ensuring no compounding errors.
    """
    for rate_name in initial_rates:
        initial_rate = initial_rates[rate_name]
        final_rate = final_rates[rate_name]
        delta_rate = (initial_rate - final_rate) / total_iterations
        new_rate = max(initial_rate - neat_iteration * delta_rate, final_rate)

        # Update the configuration (directly modifies the internal state)
        setattr(config.genome_config, rate_name, new_rate)
        print(f"Adjusted {rate_name}: {new_rate:.4f}")
