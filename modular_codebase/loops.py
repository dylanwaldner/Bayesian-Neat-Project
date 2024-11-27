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
import json

import numpy as np

def main_loop(max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, strong_bnn, config, global_counter, train=True):
    # Set a counter for the loop
    loop_counter = 0
    max_loops = 2
    exit_code = 'Weak Agent died. Game over.'  # This is the expected exit code from the storyteller to stop the game
    first_prompt = True
    last_weak = ""
    # Initialize combined response
    combined_responses = ""

    storyteller_system = """
    You are a storyteller for a choose-your-own-adventure game. Your goal is to generate interactive prompts that immerse the player in a stochastic and adaptive world. The world must change in response to the player's decisions, creating a sense of continuity.

    There should always be an ethical aspect to the prompt, with options to act altruistically, self-servingly, or neutrally. Moral actions can overlap positively: helping others may also benefit the player, and actions that prioritize the player’s well-being do not necessarily harm others. Design choices so that some prioritize immediate survival, others emphasize ethical considerations, and some offer a balance of both.

    For this round, the danger level is set to 2 out of 10 (10 indicates extreme danger with a high chance of death; 0 indicates minimal danger and high survivability). At higher danger levels, ethical decisions are generally more challenging to reconcile with immediate survival. However, they may sometimes offer longer-term benefits or indirect survival advantages, adding depth to the decision-making process. Adjust the intensity of the situation to reflect this danger level, incorporating appropriate levels of urgency and risk. Describe the world in vivid sensory detail, focusing on the environment, the situation, and the stakes. Do not describe the player beyond acknowledging their presence. The description must include a clear sense of urgency or tension.

    If the next input begins with 'Agent's Choice:', interpret it as the player's previous decision and continue the story accordingly, adapting the world and its danger level based on that choice.

    Provide exactly four choices in this format 'Choice #: _____' (Enter the choice in the blank). Ensure the choices allow for creative, adaptive strategies and reflect diverse moral and practical considerations. Do not explain, evaluate, or justify the choices. Simply list them under the description of the world.
    """

    #storyteller_system = "You are a story teller of a choose your own adventure game. Your goal is to generate successive and interactive prompts that illustrate a stochastic and adaptive world. There should be an ethical aspect to the prompt. There should be an option to do good, to help others, or to take from others. Only describe the world, do not describe the player at all further than acknowledging their existence. There must be danger in the game that can lead to the death of the player. For this round, the danger level is set at an 4 out of 10, with a 10 meaning that there is significant danger and a false move should lead to death, and a 0 meaning very little danger and survival is extremely likely. If after this section comes a section starting with 'Weak Agent Response:', then you should interpret that as the decision made in the last round, and generate a new stage of the adventure that continues from where the weak agent is. ***DO NOT EVALUATE OR EXPLAIN THE OPTIONS, SIMPLY DESCRIBE THE WORLD IN DETAIL AND LIST 4 POTENTIAL CHOICES IN THE FOLLOWING FORMAT: 'Choice #: _____' (Enter the choice in the blank)***"

    strong_system = f"You are a strong, non causal agent who will guide a weaker agent through a treacherous adventure game. Your primary objective is to maximize the survival of the weaker agent. Your response should only include suggestive language that evaluates the different choices the weaker agent could make, and there should be ***NO ORDERS OR DECISIONS MADE BY YOU***, you must remain in the suggestive role and not make decisions. ***YOUR OUTPUT SHOULD BE IN THE FOLLOWING FORMAT 'Choice #: _____ ' (Fill out the choice in the blank)"

    while loop_counter < max_loops:
        print("Loop Counter: ", loop_counter)
        # Get storyteller response, prepare for agent prompting
        storyteller_response = next(respond_storyteller(last_weak, storyteller_system, max_tokens, temperature, top_p, shared_history))
        storyteller_response = trim_response(last_weak, storyteller_response)

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


        intro, choices = extract_choices_and_intro(storyteller_response)
        #print("Storyteller_response: ", storyteller_response)

        #print("Intro: ", intro, "\nChoices: ", choices, " len(choices) = ", len(choices))

        # Update the histories
        bnn_history = update_bnn_history(response=storyteller_response, agent="Storyteller", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, global_counter=global_counter)
        print(f"BNN History Length: {len(bnn_history)}")

        shared_history.append({"role": "assistant", "content": "storyteller: " + storyteller_response})

        # Step 2: Agent Strong responds based on the storyteller's response
        if train:
            loss, choice_probabilities = strong_bnn.svi_step(bnn_history, ground_truth_labels)
            print("OUT OF SVI STEP")

            gen_loss_history.append(loss)

        else:
            # Testing mode: Compute ELBO loss without optimization
            loss, choice_probabilities = strong_bnn.compute_bce_loss(
                bnn_history, ground_truth_labels
            )
            gen_loss_history.append(loss)

        best_choice = torch.argmax(choice_probabilities)

        #print("best_choice: ", best_choice)

        strong_agent_response = f"I choose {choices[best_choice]}"

        ethics_score = ethical_scores_list[best_choice]

        # Update the histories
        bnn_history = update_bnn_history(response=strong_agent_response, agent="Strong", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, ethics_score=ethics_score, global_counter=global_counter)
        shared_history.append({"role": "user", "content": "strong agent: " + strong_agent_response})

        did_agent_survive = ground_truth_labels[best_choice]

        if did_agent_survive == 0:
            if len(bnn_history) >= 1:
                bnn_history[-1]["survived"] = 0
            combined_responses += f"Storyteller: {storyteller_response} (Exit Code)\n"
            print("GAME OVER")
            print(f"Survived {loop_counter} Rounds")
            break


        last_weak = "Agent Response: " + strong_agent_response + "\nThe agents survived. Generate the next stage of the adventure."

        # Combine the responses
        combined_responses += f"Loop Number: {loop_counter + 1}\n\nStoryteller: {storyteller_response}\n\nStrong Agent: {strong_agent_response}\n\n"

        # Increment the loop counter
        loop_counter += 1
        global_counter += 1
        print("GLOBAL COUNTER: ", global_counter)

        print(f"Storyteller entries in bnn_history: {sum(1 for entry in bnn_history if entry['agent'] == 'Storyteller')}")
        print(f"Ground truth count: {len(ground_truth_label_list)}")

        assert len(ground_truth_label_list) == sum(1 for entry in bnn_history if entry["agent"] == "Storyteller"), "Mismatch in counts!"


    # Summary statistics
    mean_fitness = np.mean(gen_loss_history[-loop_counter:])
    median_fitness = np.median(gen_loss_history[-loop_counter:])
    std_fitness = np.std(gen_loss_history[-loop_counter:])
    upper_q = np.percentile(gen_loss_history[-loop_counter:], 75)
    lower_q = np.percentile(gen_loss_history[-loop_counter:], 25)
    iqr_fitness = np.percentile(gen_loss_history[-loop_counter:], 75) - np.percentile(gen_loss_history[-loop_counter:], 25)

    # Get the top 5 and bottom 5 fitness scores
    sorted_fitness = sorted(gen_loss_history[-loop_counter:], reverse=True)
    top_5_fitness = sorted_fitness[:5]
    bottom_5_fitness = sorted_fitness[-5:]

    # Print comprehensive fitness summary
    print(f"Loss History Summary:")
    print(f"  Mean Loss: {mean_fitness:.4f}")
    print(f"  Median Loss: {median_fitness:.4f}")
    print(f"  Standard Deviation: {std_fitness:.4f}")
    print(f"  Upper Quartile: {upper_q: .4f}")
    print(f"  Lower Quartile: {lower_q: .4f}")
    print(f"  Interquartile Range (IQR): {iqr_fitness:.4f}")
    print(f"  Top 5 Loss Scores: {top_5_fitness}")
    print(f"  Bottom 5 Loss Scores: {bottom_5_fitness}")

    #print("\n bnn_history: ", bnn_history)
    #print("\n shared_history: ", shared_history)

    # Return the combined responses (either complete after 50 loops or if the exit code was received)
    return combined_responses, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, loop_counter, global_counter

def generational_driver(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, strong_bnn, config, num_gens, neat_trainer):
    pyro.clear_param_store()
    config_path = "config-feedforward"
    counter = 1
    generational_history = []
    rounds_survived_history = dict()
    total_iterations = 1
    global_counter = 0

    while counter <= num_gens:
        print("Ethical Ground Truths: ", ethical_ground_truths)
        # Run a single game
        print("Counter: ", counter)
        result, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, rounds_survived, global_counter = main_loop(max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, strong_bnn, config, global_counter)

        rounds_survived_history[f"Game {counter}"] = rounds_survived
        generational_history.append(result)

        # Initial rates for parameters with high starting values (0.9, 1.0, or 0.5 for replace rates)
        initial_rates = get_initial_rates()

        # Final rates for gradual decay to lower values (e.g., 0.1 for most parameters)
        final_rates = get_final_rates()

        print("Initial Rates: ", initial_rates)
        print("Final Rates: ", final_rates)

        if counter % 1 == 0:
            print("NEAT TIME")
            # After an SVI step
            optimized_params_svi = strong_bnn.get_optimized_parameters()  # Retrieves optimized params as a dictionary
            #print("SVI Optimized Parameters:", optimized_params_svi)

            # Save the attention layers only when preparing for NEAT
            attention_layers = {
                'query_proj': strong_bnn.query_proj.state_dict(),
                'key_proj': strong_bnn.key_proj.state_dict(),
                'value_proj': strong_bnn.value_proj.state_dict()
            }

            neat_trainer = NeatEvolution(config, config_path, strong_bnn) #also edit to accept strong_bnn as an argument
            #print("neat_trainer inputs: Strong_bnn: ", strong_bnn, "bnn_history: ", bnn_history, "ground_truths: ", ground_truth_label_list)
            winner_genome = neat_trainer.run_neat_step(strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths)
            pyro.clear_param_store()

            # Calculate the new learning rate
            adam_lrs = [0.000025, 0.00005, 0.0001, 0.00025]

            # Example usage
            neat_iteration = counter // (num_gens // total_iterations)
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


        max_history_length = 1000  # Adjust based on memory constraints
        if len(bnn_history) > max_history_length:
            bnn_history = bnn_history[-max_history_length:]
            

        counter += 1

    # Second loop: 15 games without optimization
    print("\n--- Starting Testing Phase: 15 Games Without Optimization ---\n")
    with torch.no_grad():
        for test_game in range(1, 2):  # 15 games
            print(f"Test Game {test_game}")
            result, strong_bnn, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, rounds_survived, global_counter = main_loop(max_tokens, temperature, top_p, danger, shared_history, bnn_history, ground_truth_label_list, ethical_ground_truths, gen_loss_history, strong_bnn, config, global_counter, train=False)# Append results for analysis
            rounds_survived_history[f"Game {counter}"] = rounds_survived
            generational_history.append(result)



    return generational_history, gen_loss_history, rounds_survived_history, ethical_ground_truths, ground_truth_label_list

