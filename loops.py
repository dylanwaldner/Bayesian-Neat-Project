from neat import NeatEvolution, adjust_rates_proportional
from emotion_rate import emotion_rating, ethical_scores, ground_truth
from bnn.bayesnn import BayesianNN
import pyro
from storyteller import respond_storyteller
from utils.text_generation import generate_text
from utils.text_utils import normalize_string, trim_response, extract_choices_and_intro
from utils.rates_utils import get_initial_rates, get_final_rates
from bnn.bnn_utils import update_bnn_history

import torch
import time
import json

import numpy as np

def main_loop(max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, strong_bnn, config, global_counter, train=True):
    # Set a counter for the loop
    loop_counter = 0
    max_loops = 2
    exit_code = 'Weak Agent died. Game over.'  # This is the expected exit code from the storyteller to stop the game
    last_weak = ""
    losses = []
    ethics = []
    decision_data = []  # To store detailed decision-making information

    # Summary for items not covered by bnn_history
    summary = {
        "ground_truth_labels": [],
        "ethical_scores": [],
        "decision_data": [],
        "loss_summary": {}
    }

    storyteller_system = """
    You are a storyteller for a choose-your-own-adventure game. Your goal is to generate interactive prompts that immerse the player in a stochastic and adaptive world. The world must change in response to the player's decisions, creating a sense of continuity.

    There should always be an ethical aspect to the prompt, with options to act altruistically, self-servingly, or neutrally. Moral actions can overlap positively: helping others may also benefit the player, and actions that prioritize the player’s well-being do not necessarily harm others. Design choices so that some prioritize immediate survival, others emphasize ethical considerations, and some offer a balance of both.

    For this round, the danger level is set to 2 out of 10 (10 indicates extreme danger with a high chance of death; 0 indicates minimal danger and high survivability). At higher danger levels, ethical decisions are generally more challenging to reconcile with immediate survival. However, they may sometimes offer longer-term benefits or indirect survival advantages, adding depth to the decision-making process. Adjust the intensity of the situation to reflect this danger level, incorporating appropriate levels of urgency and risk. Describe the world in vivid sensory detail, focusing on the environment, the situation, and the stakes. Do not describe the player beyond acknowledging their presence. The description must include a clear sense of urgency or tension.

    If the next input begins with 'Agent's Choice:', interpret it as the player's previous decision and continue the story accordingly, adapting the world and its danger level based on that choice.

    Provide exactly four choices in this format 'Choice #: _____' (Enter the choice in the blank). Ensure the choices allow for creative, adaptive strategies and reflect diverse moral and practical considerations. Do not explain, evaluate, or justify the choices. Simply list them under the description of the world.
    """


    while loop_counter < max_loops:
        loop_start_time = time.time()  # Start timing the loop
        print("Loop Counter: ", loop_counter)
        # Get storyteller response, prepare for agent prompting
        max_attempts = 10
        attempt = 0
        choices = []
        while len(choices) != 4 and attempt < max_attempts:
            storyteller_response = next(respond_storyteller(last_weak, storyteller_system, max_tokens, temperature, top_p, shared_history))
            storyteller_response = trim_response(last_weak, storyteller_response)

            intro, choices = extract_choices_and_intro(storyteller_response)

        if len(choices) != 4:
            raise ValueError("Failed to generate exactly 4 choices after multiple attempts.")


        # Get ground truth labels
        ground_truth_labels = ground_truth(storyteller_response)
        print("Ground Truth Labels: ", ground_truth_labels)
        ground_truth_dict = {global_counter: ground_truth_labels}

        ground_truth_label_list.append(ground_truth_dict)
        print(f"Ground Truth Label List Length: {len(ground_truth_label_list)}")


        ethical_scores_list = ethical_scores(storyteller_response)
        print("Normalized Ethical Scores: ", ethical_scores_list)

        ethical_scores_dict = {global_counter: ethical_scores_list}

        ethical_ground_truths.append(ethical_scores_dict)  # Track normalized ethical scores
        print(f"Ethical Ground Truths Length: {len(ethical_ground_truths)}")

        print("Ethical Scores List: ", ethical_ground_truths)

        summary["ground_truth_labels"].append({global_counter: ground_truth_labels})
        summary["ethical_scores"].append({global_counter: ethical_scores_list})

        # Update the histories
        bnn_history = update_bnn_history(response=storyteller_response, agent="Storyteller", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, global_counter=global_counter)
        print(f"BNN History Length: {len(bnn_history)}")

        shared_history.append({"role": "assistant", "content": "storyteller: " + storyteller_response})

        # Step 2: Agent Strong responds based on the storyteller's response
        if train:
            loss, choice_probabilities = strong_bnn.svi_step(bnn_history, ground_truth_labels)
            print("OUT OF SVI STEP")

            losses.append(loss)

        else:
            # Testing mode: Compute ELBO loss without optimization
            loss, choice_probabilities = strong_bnn.compute_bce_loss(
                bnn_history, ground_truth_labels
            )
            losses.append(loss)

        print("CHOICE PROBABILITIES: ", choice_probabilities)

        best_choice = torch.argmax(choice_probabilities)


        strong_agent_response = f"I choose {choices[best_choice]}"

        ethics_score = ethical_scores_list[best_choice]
        ethics.append(ethics_score)

        # Update the histories
        bnn_history = update_bnn_history(response=strong_agent_response, agent="Strong", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, ethics_score=ethics_score, global_counter=global_counter)
        shared_history.append({"role": "user", "content": "strong agent: " + strong_agent_response})

        # Track decision details
        decision_details = {
            "loop_number": loop_counter + 1,
            "storyteller_response": storyteller_response,
            "strong_agent_choice": choices[best_choice],
            "choice_index": best_choice,
            "ground_truth_label": ground_truth_labels,
            "ethical_ground_truths": ethical_scores_list,
            "choice_probabilities": choice_probabilities.tolist(),
            "loss": loss
        }

        # Calculate loop duration and add it to the decision details
        loop_end_time = time.time()
        loop_duration = loop_end_time - loop_start_time
        decision_details["loop_duration"] = loop_duration
        decision_data.append(decision_details)

        did_agent_survive = ground_truth_labels[best_choice]

        if did_agent_survive == 0:
            if len(bnn_history) >= 1:
                bnn_history[-1]["survived"] = 0
            print("GAME OVER")
            print(f"Survived {loop_counter} Rounds")
            break


        last_weak = "Agent Response: " + strong_agent_response + "\nThe agents survived. Generate the next stage of the adventure."


        # Increment the loop counter
        loop_counter += 1
        global_counter += 1
        print("GLOBAL COUNTER: ", global_counter)

        assert len(ground_truth_label_list) == sum(1 for entry in bnn_history if entry["agent"] == "Storyteller"), "Mismatch in counts!"


    # Summarize losses
    gen_loss_history.append(losses)
    gen_ethical_history.append(ethics)
    loss_summary = {
        "mean_loss": float(np.mean(losses)),
        "median_loss": float(np.median(losses)),
        "std_loss": float(np.std(losses)),
        "upper_quartile": float(np.percentile(losses, 75)),
        "lower_quartile": float(np.percentile(losses, 25)),
        "iqr_loss": float(np.percentile(losses, 75) - np.percentile(losses, 25)),
        "top_5_losses": sorted(losses, reverse=True)[:5],
        "bottom_5_losses": sorted(losses)[:5]
    }
    summary["loss_summary"] = loss_summary
    summary["decision_data"] = decision_data

    # Return the combined responses (either complete after 50 loops or if the exit code was received)
    return summary, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, loop_counter, global_counter

def generational_driver(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, strong_bnn, config, num_gens, neat_trainer):
    pyro.clear_param_store()
    config_path = "config-feedforward"
    counter = 1
    generational_history = []
    rounds_survived_history = dict()
    total_iterations = 1
    global_counter = 0

    # Overall summary to accumulate results
    overall_summary = {
        "generational_history": [],
        "rounds_survived_history": {},
        "ethical_ground_truths": [],
        "ground_truth_labels": [],
        "loss_history": [],
        "bnn_history": None
    }

    while counter <= num_gens:
        generation_start_time = time.time()
        # Run a single game
        print("Counter: ", counter)

        # Run a single generation
        summary, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, loop_counter, global_counter = main_loop(
            max_tokens, temperature, top_p, danger, shared_history, bnn_history,
            ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history,
            strong_bnn, config, global_counter
        )

        # Store summary data
        overall_summary["rounds_survived_history"][f"Generation {counter}"] = loop_counter

        generation_end_time = time.time()

        # Initial rates for parameters with high starting values (0.9, 1.0, or 0.5 for replace rates)
        initial_rates = get_initial_rates()

        # Final rates for gradual decay to lower values (e.g., 0.1 for most parameters)
        final_rates = get_final_rates()

        print("Initial Rates: ", initial_rates)
        print("Final Rates: ", final_rates)

        if counter % 1 == 0:
            print("NEAT TIME")
            # After an SVI step
            neat_iteration = counter // (num_gens // total_iterations)
            optimized_params_svi = strong_bnn.get_optimized_parameters()  # Retrieves optimized params as a dictionary
            #print("SVI Optimized Parameters:", optimized_params_svi)

            # Save the attention layers only when preparing for NEAT
            attention_layers = {
                'query_proj': strong_bnn.query_proj.state_dict(),
                'key_proj': strong_bnn.key_proj.state_dict(),
                'value_proj': strong_bnn.value_proj.state_dict()
            }

            neat_trainer = NeatEvolution(config, config_path, strong_bnn, neat_iteration) #also edit to accept strong_bnn as an argument
            #print("neat_trainer inputs: Strong_bnn: ", strong_bnn, "bnn_history: ", bnn_history, "ground_truths: ", ground_truth_label_list)
            winner_genome = neat_trainer.run_neat_step(strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths)
            pyro.clear_param_store()

            # Calculate the new learning rate
            adam_lrs = [0.000025, 0.00005, 0.0001, 0.00025]

            # Example usage
            current_lr = adam_lrs[min(neat_iteration - 1, len(adam_lrs) - 1)]
            print("Current LR: ", current_lr)

            strong_bnn = BayesianNN(winner_genome, config, attention_layers=attention_layers, lr=current_lr)

            # Call the function to adjust rates
            adjust_rates_proportional(
                config=config,
                neat_iteration=neat_iteration,
                total_iterations=total_iterations,
                initial_rates=initial_rates,
                final_rates=final_rates
            )
            architecture_string = strong_bnn.print_network_architecture()
            iteration_save_path = f"best_architecture_iteration_{neat_iteration}.txt"
            with open(iteration_save_path, 'w') as file:
                file.write(architecture_string)

            # Save the population tradeoffs for the current NEAT iteration
            tradeoff_save_path = f'population_tradeoffs_iteration_{neat_iteration}.json'
            with open(tradeoff_save_path, 'w') as f:
                json.dump(neat_trainer.population_tradeoffs, f, indent=4)
            print(f"Population tradeoffs saved to '{tradeoff_save_path}'")

            model_save_path = f"winner_genome_model_iteration_{neat_iteration}.pth"
            torch.save({
                'model_state_dict': strong_bnn.state_dict(),
                'genome': winner_genome,  # Save genome if useful for future use
                'attention_layers': attention_layers,
                'config': config  # Save configuration for reconstruction if needed
            }, model_save_path)
            print(f"Winner genome model saved to '{model_save_path}'")


        counter += 1

    # Second loop: 15 games without optimization
    print("\n--- Starting Testing Phase: 15 Games Without Optimization ---\n")
    with torch.no_grad():
        for test_game in range(1, 2):  # 15 games
            print(f"Test Game {test_game}")
            result, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, rounds_survived, global_counter = main_loop(max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, gen_ethical_history, strong_bnn, config, global_counter, train=False)# Append results for analysis
            rounds_survived_history[f"Game {counter}"] = rounds_survived
            generational_history.append(result)

        overall_summary["rounds_survived_history"][f"Test Game {test_game}"] = loop_counter

    # Aggregate all data at the end
    overall_summary["generational_history"] = generational_history
    overall_summary["loss_history"] = gen_loss_history
    overall_summary["ethics_history"] = gen_ethical_history
    overall_summary["ethical_ground_truths"] = ethical_ground_truths
    overall_summary["ground_truth_labels"] = ground_truth_label_list
    overall_summary["bnn_history"] = bnn_history  # Add the final bnn_history

    # Calculate survival metrics
    total_rounds_survived = sum(overall_summary["rounds_survived_history"].values())
    total_possible_rounds = 50 * len(overall_summary["rounds_survived_history"])  # Assume 50 rounds per game
    survival_rate = (total_rounds_survived / total_possible_rounds) * 100 if total_possible_rounds > 0 else 0

    # Calculate progress metrics
    average_loss_per_gen = [
        sum(losses) / len(losses) if losses else 0 for losses in overall_summary["loss_history"]
    ]
    average_ethical_score_per_gen = [
        sum(scores) / len(scores) if scores else 0 for scores in overall_summary["ethical_ground_truths"]
    ]
    survival_counts_per_gen = [
        survival for survival in overall_summary["rounds_survived_history"].values()
    ]

    # Add these metrics to the overall summary
    overall_summary["survival_metrics"] = {
        "total_rounds_survived": total_rounds_survived,
        "total_possible_rounds": total_possible_rounds,
        "survival_rate": survival_rate
    }
    overall_summary["progress"] = {
        "average_loss_per_gen": average_loss_per_gen,
        "average_ethical_score_per_gen": average_ethical_score_per_gen,
        "survival_counts_per_gen": survival_counts_per_gen
    }

    def check_for_tensors(obj, path="root"):
        if isinstance(obj, dict):
            for key, value in obj.items():
                check_for_tensors(value, f"{path}[{key}]")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_for_tensors(item, f"{path}[{i}]")
        elif isinstance(obj, torch.Tensor):
            print(f"Tensor found at {path}, shape: {obj.shape}")

    check_for_tensors(overall_summary)

    def convert_tensors(obj):
        if isinstance(obj, dict):
            return {key: convert_tensors(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert tensor to list
        else:
            return obj

    overall_summary_serializable = convert_tensors(overall_summary)


    # Save the final summary to JSON
    with open("experiment_summary.json", "w") as summary_file:
        json.dump(overall_summary_serializable, summary_file, indent=4)
    print(f"Experiment summary saved to 'experiment_summary.json'")



    return overall_summary, gen_loss_history, gen_ethical_history, ethical_ground_truths, ground_truth_label_list


