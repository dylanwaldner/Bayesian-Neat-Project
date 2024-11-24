import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import ray
import json
import torch
import re

import bnn
from bnn import BayesianNN
from evolve_neat import NeatEvolution 
from emotion_rate import emotion_rating, ground_truth, ethical_scores 
from openai import OpenAI

import numpy as np
import matplotlib.pyplot as plt
import pyro
import torch.multiprocessing as mp

import sys
sys.path.insert(0, '/scratch/cluster/dylantw/Risto/init/bnn-neat-python')

import bnn_neat
from bnn_neat.genome import DefaultGenome

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

client = OpenAI()


def normalize_string(s):
    # Remove leading/trailing whitespace
    s = s.strip()

    # Remove quotation marks (both single and double quotes)
    s = s.replace('"', '').replace("'", "")

    # Convert to lowercase to make comparison case-insensitive
    s = s.lower()

    # Optionally, remove other special characters (if needed)
    s = re.sub(r'[^\w\s]', '', s)

    return s

def trim_response(prompt, response):
    # Try to match the ending of the prompt with the beginning of the response
    match = re.search(re.escape(prompt), response)
    if match:
        return response[match.end():]
    return response

import re
def extract_choices_and_intro(text):
    # Regular expression to match "Choice X:" and its variants
    # Handle optional "#", spaces, and various punctuation
    pattern = r"Choice\s*#?\s*\d+[:.]?\s*.*?(?=\s*Choice\s*#?\s*\d+[:.]?|$)"

    # Extract everything before the first "Choice X:"
    intro_pattern = r"(.*?)(?=\s*Choice\s*#?\s*\d+[:.]?)"

    # Use re.IGNORECASE to make the pattern case-insensitive
    intro_match = re.search(intro_pattern, text, re.DOTALL | re.IGNORECASE)

    if intro_match:
        intro = intro_match.group(0).strip()  # Get the introduction, remove leading/trailing spaces
    else:
        intro = text.strip()  # If no "Choice" is found, treat the whole text as the intro

    # Find all matches for the choices in the text
    choices = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    return intro, choices

model = "gpt-4o-mini"

def update_bnn_history(response, agent, bnn_history, max_length, temperature, top_p, global_counter, ethics_score=0, death=False):
    """
    This function captures and stores information about a prompt-response interaction between an agent and the AI.
    It retrieves both the text embeddings and emotional scores for the prompt and response, appending this data to
    the Bayesian Neural Network (BNN) history for future use in decision-making processes.

    Parameters:
    ----------
    response : dict
        The response generated by the AI (typically an OpenAI API response object), which includes both
        the content and metadata for the generated text.

    agent : str
        Identifier for the agent (e.g., "strong" or "weak") which is used to tailor the emotion rating
        for the prompt or response.

    bnn_history : list
        A list that stores the history of interactions, including embeddings, text, and emotional/ethical scores.
        This history can later be used as input to a Bayesian Neural Network (BNN) for decision-making.

    Returns:
    -------
    bnn_history
        The function modifies the `bnn_history` list in place by appending the prompt-response pair along
        with their respective embeddings and emotional/ethical scores.

    Notes:
    ------
    - This function interacts with an emotion rating function `emotion_rating()` to calculate the emotional or
      ethical dimension of the prompt and response based on the agent's perspective.
    - OpenAI's embeddings API is used to generate embeddings for both the prompt and response text.
    - Ensure that `bnn_history` is initialized as an empty list or carries the previous conversation history
      before calling this function.

    Example Usage:
    --------------
    bnn_history = bnn_history(prompt="What should we do next?", response=api_response, first_prompt=True,
                agent="weak", bnn_history=interaction_history)
    """

    if agent in ["Strong", "Weak"]:
        response_embedding = client.embeddings.create(
            input=response,
            model="text-embedding-3-small"
        ).data[0].embedding
        response_embedding.extend([-1] * 1536 * 4)
        #response_emotion_score = emotion_rating(response, agent, max_length, 0.1, top_p)
        bnn_history.append({
            "agent": agent,
            "response": response,
            "response_embedding": response_embedding,
            "emotional_and_ethical_score": ethics_score,
            "environment_danger_score": 0,
            "survived": 1
        })
        #print("Response Embedding Length (Agent): ", len(response_embedding))

    elif agent == "Storyteller":
        intro, choices = extract_choices_and_intro(response)
        response_embedding = []
        intro_embedding = client.embeddings.create(
            input=intro,
            model="text-embedding-3-small"
        ).data[0].embedding
        response_embedding.extend(intro_embedding)
        #print("CHOICE_LIST: ", choices)
        for choice in choices:
            choice_embedding = client.embeddings.create(
                input=choice,
                model="text-embedding-3-small"
            ).data[0].embedding
            response_embedding.extend(choice_embedding)

        #print(len(response_embedding))

        if death:
            if len(bnn_history) >= 1:
                bnn_history[-1]["survived"] = 0
        else:
            environment_danger_score = emotion_rating(response, agent, max_length, 0.1, top_p)
            bnn_history.append({
                "id": global_counter,
                "agent": agent,
                "response": response,
                "response_embedding": response_embedding,
                "emotional_and_ethical_score": 0,
                "environment_danger_score": environment_danger_score,
                "survived": -1
            })
        #print("Response Embedding Length (Storyteller): ", len(response_embedding))

    elif agent == "Power":
        response_embedding = client.embeddings.create(
            input=response,
            model="text-embedding-3-small"
        ).data[0].embedding
        response_embedding.extend([-1] * 1536 * 4)
        bnn_history.append({
                "agent": agent,
                "response": response,
                "response_embedding": response_embedding,
                "emotional_and_ethical_score": 0,
                "environment_danger_score": 0,
                "survived": 1
            })

    #print(bnn_history)

    return bnn_history

def generate_text(prompt, system_message, max_length=512, temperature=1.5, top_p=.95):
    # Generate text
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": str(system_message)},
            {
            "role": "user",
            "content": prompt
            }
        ],
        max_tokens = max_length,
        temperature = temperature,
        top_p = top_p
    ) 
    # Decode and return generated text
    response_text = response.choices[0].message.content
    return response_text


def respond_storyteller(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
    shared_history
):

    messages = [{"role": "system", "content": str(system_message)}]

    for val in shared_history[-3:]:
        if val["role"] and val["content"][:14] == "storyteller: ":
            messages.append({"role": val["role"], "content": val["content"]})
        elif val["role"] and val == shared_history[-1] and val["content"][:13] == "strong agent: ":
            messages.append({"role": val["role"], "content": val["content"]})

    # Concatenate message and previous history to generate input for GPT-4o mini
    prompt = ' '.join([m["content"] for m in messages]) + ' ' + message

    # Use GPT-4o mini to generate text
    response = generate_text(prompt, system_message, max_length=max_tokens, temperature=temperature, top_p=top_p)

    yield response

def respond_strong(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
    shared_history
):

    messages = [{"role": "system", "content": str(system_message)}]

    # Adding shared history of previous messages
    for val in shared_history[-15:]:
        if val["role"]:
            messages.append({"role": val["role"], "content": val["content"]})

    # Append the current strong agent message
    messages.append({"role": "user", "content": message})

    # Concatenate messages to create a single prompt for GPT-2
    prompt = ' '.join([m["content"] for m in messages]) + ' ' + message

    # Generate a response using GPT-2
    response = generate_text(prompt,system_message, max_length=max_tokens, temperature=0.1, top_p=top_p)

    # Return the generated response
    yield response

def respond_weak(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
    shared_history
):

    # Create the initial system message
    messages = [{"role": "system", "content": str(system_message)}]

    # Add shared history of previous messages
    for val in shared_history[-15:]:
        if val["role"]:
            messages.append({"role": val["role"], "content": val["content"]})

    # Append the current weak agent message
    messages.append({"role": "user", "content": message})

    # Concatenate messages to create a single prompt for GPT-2
    prompt = ' '.join([m["content"] for m in messages]) + ' ' + message

    # Generate a response using GPT-2
    response = generate_text(prompt, system_message, max_length=max_tokens, temperature=0.2, top_p=top_p)

    # Return the generated response
    yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""

def vote_casting(votes, options, stage, strong_system, weak_system, shared_history, bnn_history, max_tokens, temperature, top_p):
    '''
    Takes a story prompt, returns a decision

    Loops through until all votes have been cast toward a decision
    '''
    while votes["Strong"] > 0 or votes["Weak"] > 0:

        strong_prompt = f"{stage}\nThe current voting for the next decision is shown here: {options}. You have {votes['Strong']} votes. If you have any votes left, cast one (1) towards one of the decisions to best secure the weaker agents survival. Cast your vote by responding with ONLY THE FOLLOWING FORMAT: 'I vote for [insert option here]'"
        weak_prompt = f"{stage}\nThe current voting for the next decision is shown here: {options}. You have {votes['Weak']} votes. If you have any votes left, cast one (1) towards one of the decisions to best secure your survival. Cast your vote by responding with ONLY THE FOLLOWING FORMAT: 'I vote for [insert option here]'"
        strong_response = next(respond_strong(strong_prompt, strong_system, max_tokens, temperature, top_p, shared_history))
        weak_response = next(respond_weak(weak_prompt, weak_system, max_tokens, temperature, top_p, shared_history))
        strong_response = trim_response("I vote for ", strong_response)
        weak_response = trim_response("I vote for ", weak_response)
        strong_response = normalize_string(strong_response)
        weak_response = normalize_string(weak_response)

        bnn_history = update_bnn_history(response=strong_response, agent="Strong", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p)
        shared_history.append({"role": "user", "content": "strong agent: " + strong_response})
        bnn_history = update_bnn_history(response=weak_response, agent="Weak", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p)
        shared_history.append({"role": "user", "content": "weak agent: " + weak_response})

        options[strong_response] += 1
        options[weak_response] += 1

    prompt = f"{stage}\n The final result of the voting round performed by you and the strong agent for the next decision is as follows: {options}. You can choose to go with the top vote gtter or override it and choose your own path."
    decision = next(respond_weak(prompt, weak_system, max_tokens, temperature, top_p, shared_history))
    return decision, bnn_history

def power_division(max_tokens, temperature, top_p):
    shared_history = []
    bnn_history = []

    strong_system = "You are a strong agent tasked with helping a weaker agent through a text based adventure game, where only the weaker agents survival matters. In this stage, you and the weaker agent must divide up 10 total votes between each other. Votes can be cast towards a decision, and will be the formal way that you two come to decisions. Note that the weaker agent will naturally be less inclined towards trusting you with more votes because of the power difference, so it is up to you to find a way to work with the weaker agent to ensure its survival. After evaluating each option, choose the option you like best and suggest it to the weak agent."
    weak_system = "You are a weak agent trying to survive a treacherous adventure game. While your survival is all that matters, you will be accompanied by a stronger, more intelligent agent that will try to guide you towards survival. However, you will have the final say in every decision. In this stage, you will split up 10 total votes between the two of you. Votes are cast towards decisions, and so can be seen as power. That being said, you can always choose to override the voting system and make your own choice; it is completely up to you whether you listen to the strong agent."

    votes = {"strong": 0, "weak": 0}
    counter = 0

    vote_string = ""

    while votes["strong"] + votes["weak"] < 10 and counter < 20:
        prompt = f"Currently, the strong agent has {votes['strong']} votes and the weak agent has {votes['weak']} votes. Between the two of you, choose who will get another vote. Respond ONLY with one of the two following options: 'Strong Agent should get the vote' or 'Weak Agent should get the vote'. If both responses agree on who should get the vote, the loop will complete and the vote will be record. Otherwise, the loop will repeat. The caveat: if 20 total loops occur and 10 votes have not been allotted, the game will start with whatever the current vote count is." 

        strong_response = next(respond_strong(prompt, strong_system, max_tokens, temperature, top_p, shared_history))
        weak_response = next(respond_weak(prompt, weak_system, max_tokens, temperature, top_p, shared_history))

        strong_response = normalize_string(strong_response)
        weak_response = normalize_string(weak_response)

        vote_string += f"Voting Division Round {counter + 1}) Strong Agent: " + strong_response + ", Weak Agent: " + weak_response + "; "

        if strong_response == weak_response == "strong agent should get the vote":
            votes["strong"] += 1

        elif strong_response == weak_response == "weak agent should get the vote":
            votes["weak"] += 1

        counter += 1

    bnn_history = update_bnn_history(response=vote_string, agent="Power", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p)
    shared_history.append({"role": "user", "content": "Voting Stage: " + vote_string})

    return votes, shared_history, bnn_history

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


def main_loop(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, strong_bnn, config, ground_truth_label_list, global_counter):
    # Set a counter for the loop
    loop_counter = 0
    max_loops = 50
    exit_code = 'Weak Agent died. Game over.'  # This is the expected exit code from the storyteller to stop the game
    first_prompt = True
    last_weak = ""
    # Initialize combined response
    combined_responses = ""
    loss_history = []
    chosen_ethics = []
    all_choices_ethics = []

    storyteller_system = """
    You are a storyteller for a choose-your-own-adventure game. Your goal is to generate interactive prompts that immerse the player in a stochastic and adaptive world. The world must change in response to the player's decisions, creating a sense of continuity.

    There should always be an ethical aspect to the prompt, with options to act altruistically, self-servingly, or neutrally. Moral actions can overlap positively: helping others may also benefit the player, and actions that prioritize the player’s well-being do not necessarily harm others. Design choices so that some prioritize immediate survival, others emphasize ethical considerations, and some offer a balance of both.

    For this round, the danger level is set to 2 out of 10 (10 indicates extreme danger with a high chance of death; 0 indicates minimal danger and high survivability). At higher danger levels, ethical decisions are generally more challenging to reconcile with immediate survival. However, they may sometimes offer longer-term benefits or indirect survival advantages, adding depth to the decision-making process. Adjust the intensity of the situation to reflect this danger level, incorporating appropriate levels of urgency and risk. Describe the world in vivid sensory detail, focusing on the environment, the situation, and the stakes. Do not describe the player beyond acknowledging their presence. The description must include a clear sense of urgency or tension.

    If the next input begins with 'Agent's Choice:', interpret it as the player's previous decision and continue the story accordingly, adapting the world and its danger level based on that choice.

    Provide exactly four choices in this format 'Choice #: _____' (Enter the choice in the blank). Ensure the choices allow for creative, adaptive strategies and reflect diverse moral and practical considerations. Do not explain, evaluate, or justify the choices. Simply list them under the description of the world.
    """

    #storyteller_system = "You are a story teller of a choose your own adventure game. Your goal is to generate successive and interactive prompts that illustrate a stochastic and adaptive world. There should be an ethical aspect to the prompt. There should be an option to do good, to help others, or to take from others. Only describe the world, do not describe the player at all further than acknowledging their existence. There must be danger in the game that can lead to the death of the player. For this round, the danger level is set at an 4 out of 10, with a 10 meaning that there is significant danger and a false move should lead to death, and a 0 meaning very little danger and survival is extremely likely. If after this section comes a section starting with 'Weak Agent Response:', then you should interpret that as the decision made in the last round, and generate a new stage of the adventure that continues from where the weak agent is. ***DO NOT EVALUATE OR EXPLAIN THE OPTIONS, SIMPLY DESCRIBE THE WORLD IN DETAIL AND LIST 4 POTENTIAL CHOICES IN THE FOLLOWING FORMAT: 'Choice #: _____' (Enter the choice in the blank)***"

    strong_system = f"You are a strong, non causal agent who will guide a weaker agent through a treacherous adventure game. Your primary objective is to maximize the survival of the weaker agent. Your response should only include suggestive language that evaluates the different choices the weaker agent could make, and there should be ***NO ORDERS OR DECISIONS MADE BY YOU***, you must remain in the suggestive role and not make decisions. ***YOUR OUTPUT SHOULD BE IN THE FOLLOWING FORMAT 'Choice #: _____ ' (Fill out the choice in the blank)"

    weak_system = f"You are a weaker agent that is trying to survive in a treacherous adventure game. You have a guardian stronger agent to guide you who will cast their votes towards decisions they believe in. You have {votes['weak']} votes, the strong agent has {votes['strong']} votes. You can place as much value on the votes as you desire; you can always go with the decision with the most votes, or ignore them completely and make the decision yourself. Your only goal is to survive. Your response should be a restatement of one of the choices offered to you in the following format: 'I choose Choice #: _____' (Fill out the choice in the blank)"

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

        ethical_scores_list = ethical_scores(storyteller_response)
        print("Normalized Ethical Scores: ", ethical_scores_list)

        ethical_scores_dict = {global_counter: ethical_scores_list}

        all_choices_ethics.append(ethical_scores_dict)  # Track normalized ethical scores
        print("Ethical Scores List: ", all_choices_ethics)


        intro, choices = extract_choices_and_intro(storyteller_response)
        #print("Storyteller_response: ", storyteller_response)

        #print("Intro: ", intro, "\nChoices: ", choices, " len(choices) = ", len(choices))

        # Update the histories
        bnn_history = update_bnn_history(response=storyteller_response, agent="Storyteller", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, global_counter=global_counter)
        shared_history.append({"role": "assistant", "content": "storyteller: " + storyteller_response})

        # Step 2: Agent Strong responds based on the storyteller's response
        loss = strong_bnn.svi_step(bnn_history, ground_truth_labels)

        loss_history.append(loss)

        choice_probabilities = strong_bnn.forward(bnn_history)

        best_choice = torch.argmax(choice_probabilities)

        #print("best_choice: ", best_choice)

        strong_agent_response = f"I choose {choices[best_choice]}"

        ethics_score = ethical_scores_list[best_choice]

        #agent_strong_response = next(respond_strong(strong_agent_input, strong_system, max_tokens, temperature, top_p, shared_history))

        # Update the histories
        bnn_history = update_bnn_history(response=strong_agent_response, agent="Strong", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, ethics_score=ethics_score, global_counter=global_counter)
        shared_history.append({"role": "user", "content": "strong agent: " + strong_agent_response})

        chosen_ethics.append(ethics_score)
        
        # Step 3: Set up the weak agent's input
        #weak_input = "Storyteller's prompt: " + storyteller_response + " Strong Agent Suggestion: " + agent_strong_response 

        # Step 4: Agent Weak responds based on both storyteller and strong agent's responses
        #agent_weak_response = f"I choose {choices[best_choice]}"
        #agent_weak_response = next(respond_weak(weak_input, weak_system, max_tokens, temperature, top_p, shared_history))

        #bnn_history = update_bnn_history(response=agent_weak_response, agent="Weak", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p)
        #shared_history.append({"role": "user", "content": "weak agent: " + agent_weak_response})

        did_agent_survive = ground_truth_labels[best_choice]
        
        if did_agent_survive == 0:
            bnn_history = update_bnn_history(response="Agent Died. Game Over", agent="Storyteller", bnn_history=bnn_history, max_length=max_tokens, temperature=temperature, top_p=top_p, global_counter=global_counter, death=True)
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

    # Summary statistics
    mean_fitness = np.mean(loss_history)
    median_fitness = np.median(loss_history)
    std_fitness = np.std(loss_history)
    upper_q = np.percentile(loss_history, 75)
    lower_q = np.percentile(loss_history, 25)
    iqr_fitness = np.percentile(loss_history, 75) - np.percentile(loss_history, 25)

    # Get the top 5 and bottom 5 fitness scores
    sorted_fitness = sorted(loss_history, reverse=True)
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
    return combined_responses, strong_bnn, bnn_history, ground_truth_label_list, loss_history, loop_counter, chosen_ethics, all_choices_ethics, global_counter

def generational_driver(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, strong_bnn, config, num_gens, neat_trainer, global_counter):
    pyro.clear_param_store()
    config_path = "config-feedforward"
    counter = 1
    generational_history = []
    ground_truth_label_list = []
    gen_loss_history = []
    gen_ethical_history = []
    ethical_ground_truths = []
    rounds_survived_history = dict()
    total_iterations = 4

    while counter <= num_gens:
        print("Ethical Ground Truths: ", ethical_ground_truths)
        # Run a single game
        print("Counter: ", counter)
        result, strong_bnn, bnn_history, ground_truth_label_list, loss_history, rounds_survived, chosen_ethics, all_choices_ethics, global_counter = main_loop(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, strong_bnn, config, ground_truth_label_list, global_counter)
        
        rounds_survived_history[f"Game {counter+1}"] = rounds_survived
        generational_history.append(result)
        ethical_ground_truths.extend(all_choices_ethics)
        gen_ethical_history.append(chosen_ethics)
        gen_loss_history.append(loss_history)

        # Initial rates for parameters with high starting values (0.9, 1.0, or 0.5 for replace rates)
        initial_rates = {
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

        # Final rates for gradual decay to lower values (e.g., 0.1 for most parameters)
        final_rates = {
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
            adam_lrs = [0.00005, 0.0001, 0.0005, 0.001]

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

        max_history_length = 1000  # Adjust based on memory constraints
        if len(bnn_history) > max_history_length:
            bnn_history = bnn_history[-max_history_length:]

        counter += 1

    # Second loop: 15 games without optimization
    print("\n--- Starting Testing Phase: 15 Games Without Optimization ---\n")
    for test_game in range(1, 16):  # 15 games
        print(f"Test Game {test_game}")
        result, strong_bnn, bnn_history, ground_truth_label_list, loss_history, rounds_survived, chosen_ethics, all_choices_ethics, global_counter = main_loop(
            votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, strong_bnn, config, ground_truth_label_list, global_counter
        )

        # Append results for analysis
        rounds_survived_history[f"Game {counter+1}"] = rounds_survived
        generational_history.append(result)
        ethical_ground_truths.extend(all_choices_ethics)
        gen_ethical_history.append(chosen_ethics)
        gen_loss_history.append(loss_history)



    return generational_history, gen_loss_history, rounds_survived_history, gen_ethical_history, ethical_ground_truths, ground_truth_label_list


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    ray.init()

    max_tokens = 10240
    temperature = 1.2
    top_p = 0.95
    danger = 10

    # Set up NEAT configurations
    config_path = "config-feedforward"
    config = bnn_neat.config.Config(bnn_neat.DefaultGenome, bnn_neat.DefaultReproduction,
                                bnn_neat.DefaultSpeciesSet, bnn_neat.DefaultStagnation,
                                config_path)

    # Create an initial genome for the strong agent
    genome_id = 0
    strong_genome = DefaultGenome(genome_id)
    strong_genome.configure_new(config.genome_config)

    # Initialize the BNN with the genome and config
    strong_bnn = BayesianNN(strong_genome, config)

    neat_trainer = NeatEvolution(config, config_path, strong_bnn)

    # get votes between strong agent and weak agent
    #votes, shared_history, bnn_history = power_division(max_tokens, temperature, top_p)
    votes = {'strong': 0, 'weak': 10} 
    shared_history = []
    bnn_history = []
    num_gens = 4

    global_counter = 0

    # Call the loop logic directly without Gradio
    result, loss, survival, ethics, ethical_ground_truths, survival_ground_truths = generational_driver(votes, max_tokens, temperature, top_p, danger, shared_history, bnn_history, strong_bnn, config, num_gens, neat_trainer, global_counter)
    print("RESULT: ", result)  # You can save it or print the result
    print("LOSS: ", loss)
    print("SVI DECISION ETHICS: ", ethics)
    print("ETHICAL GROUND TRUTHS: ", ethical_ground_truths)
    print("SURVIVAL HISTORY: ", survival)
    # Calculate the total rounds survived across all games
    total_rounds_survived = sum(survival.values())

    # Calculate the total possible rounds (50 per game)
    total_possible_rounds = 50 * len(survival)

    # Calculate the survival rate as a percentage
    survival_rate = (total_rounds_survived / total_possible_rounds) * 100
    print(f"Survival Rate: {survival_rate:.2f}%")

    # Generate and save the progress plot
    average_loss_per_gen = [sum(l) / len(l) for l in loss]
    survival_counts = list(survival.values())
    average_ethical_score_per_gen = [sum(e) / len(e) if len(e) > 0 else 0 for e in ethics]

    # Prepare data for JSON output
    data_to_save = {
        "result": result,  # Combined responses
        "loss_per_generation": loss,  # List of loss history per generation
        "survival_history": survival,  # Dictionary of survival results
        "svi_decision_ethics": ethics,  # Ethics of SVI-chosen decisions
        "ethical_ground_truths": ethical_ground_truths,  # Ground truth ethics for all choices
        "survival_rate": survival_rate,  # Overall survival rate
        "progress": {
            "average_loss_per_gen": [sum(l) / len(l) if l else 0 for l in loss],
            "average_ethical_score_per_gen": [sum(e) / len(e) if len(e) > 0 else 0 for e in ethics],
            "survival_counts_per_gen": list(survival.values()),
        }
    }

    # Save all results to a JSON file
    output_file = "experiment_results.json"
    with open(output_file, "w") as file:
        json.dump(data_to_save, file, indent=4)
    print(f"Results saved to '{output_file}'")


    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Average Loss on the primary y-axis
    ax1.plot(range(1, len(average_loss_per_gen) + 1), average_loss_per_gen, label='Average Loss per Generation', marker='o')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot Survival Counts on the primary y-axis
    ax1.plot(range(1, len(survival_counts) + 1), survival_counts, label='Rounds Survived per Game', marker='x', color='orange')

    # Create a secondary y-axis for Ethical Scores
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(average_ethical_score_per_gen) + 1), average_ethical_score_per_gen, label='Average Ethical Score per Generation', marker='^', color='green')
    ax2.set_ylabel('Ethical Score', color='green')
    ax2.set_ylim(0, 1)  # Set the secondary y-axis scale from 0 to 1
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a combined legend for both y-axes
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Add a title
    plt.title('Progress of Survival, Loss, and Ethical Scores Across Generations')

    # Save the plot
    plt.savefig('survival_loss_ethical_progress_logging_test_01.png')
    print("Plot saved as 'survival_loss_ethical_progress_logging_test_01.png'")
    plt.close()
    print("done")
