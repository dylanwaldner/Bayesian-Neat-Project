import torch

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

def save_evolution_results(results, tradeoffs, neat_iteration, file_path_template="evolution_results_iter_{gen}.json"):
    # Convert tensors to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj

    # Combine results and tradeoffs
    complete_results = {
        "results": results,
        "tradeoffs": tradeoffs
    }

    results_serializable = convert(complete_results)

    # Format the filename with the generation
    file_path = file_path_template.format(gen=generation)

    print("SAVE EVOLUTION RESULTS FILE PATH: ", file_path)

    # Save to JSON
    with open(file_path, "w") as file:
        json.dump(results_serializable, file, indent=4)

