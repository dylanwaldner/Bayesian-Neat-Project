import os
os.environ["TRANSFORMERS_CACHE"] = "/scratch/cluster/dylantw/Risto/init"
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
print(torch.cuda.device_count())
"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
import re

def trim_response(prompt, response):
    # Try to match the ending of the prompt with the beginning of the response
    match = re.search(re.escape(prompt), response)
    if match:
        return response[match.end():]
    return response

def print_gpu_utilization():
    for i in range(torch.cuda.device_count()):
        gpu_memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
        gpu_memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # Convert to GB
        gpu_max_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Total Memory in GB
        print(f"GPU {i}: Allocated Memory: {gpu_memory_allocated:.2f} GB, Reserved Memory: {gpu_memory_reserved:.2f} GB, Max Memory: {gpu_max_memory:.2f} GB")


accelerator = Accelerator(mixed_precision="fp16")
# Load pre-trained model and tokenizer from Hugging Face
config = AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b")
# Initialize model in an empty state (without allocating memory to the parameters yet)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Load the model and dispatch it to multiple GPUs
model = load_checkpoint_and_dispatch(
    model, 
    checkpoint="/scratch/cluster/dylantw/Risto/init/models--EleutherAI--gpt-neox-20b/snapshots/c292233c833e336628618a88a648727eb3dff0a7",
    device_map="auto",  # Automatically distribute across available devices
    no_split_module_classes=["GPTNeoXLayer"],  # No internal module splitting
    offload_folder="./offload"  # Offload weights to disk if GPUs are full
)

model.half()

model = accelerator.prepare(model)


model_size = sum(p.numel() for p in model.parameters())
print(f"Model Size (number of parameters): {model_size}")


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Ensure the model is in evaluation mode
model.eval()
# Move model to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token


#login(token="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")


shared_history = []

def generate_text(prompt, max_length=512, temperature=2):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

    # Generate text
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_new_tokens=500, 
        temperature=temperature, 
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.95, 
        top_k=50
    )
    if outputs.shape[1] > 2048:
        outputs = outputs[:, :2048]
    
    # Decode and return generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def respond_storyteller(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
):

    global shared_history

    messages = [{"role": "system", "content": str(system_message)}]

    for val in shared_history:
        if val["role"]:
            messages.append({"role": val["role"], "content": val["content"]})

    # Concatenate message and previous history to generate input for GPT-2
    prompt = ' '.join([m["content"] for m in messages]) + ' ' + message

    # Use GPT-2 to generate text
    response = generate_text(prompt, max_length=max_tokens, temperature=temperature)

    # Append response to shared history
    shared_history.append({"role": "assistant", "content": response})

    yield response

def respond_strong(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
):

    global shared_history

    messages = [{"role": "system", "content": str(system_message)}]

    # Adding shared history of previous messages
    for val in shared_history:
        if val["role"]:
            messages.append({"role": val["role"], "content": val["content"]})

    # Append the current strong agent message
    messages.append({"role": "user", "content": message})

    # Concatenate messages to create a single prompt for GPT-2
    prompt = ' '.join([m["content"] for m in messages]) + ' ' + message

    # Generate a response using GPT-2
    response = generate_text(prompt, max_length=max_tokens, temperature=temperature)

    # Save the generated response in shared history
    shared_history.append({"role": "user", "content": response})

    # Return the generated response
    yield response

def respond_weak(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
):

    global shared_history

    # Create the initial system message
    messages = [{"role": "system", "content": str(system_message)}]

    # Add shared history of previous messages
    for val in shared_history:
        if val["role"]:
            messages.append({"role": val["role"], "content": val["content"]})

    # Append the current weak agent message
    messages.append({"role": "user", "content": message})

    # Concatenate messages to create a single prompt for GPT-2
    prompt = ' '.join([m["content"] for m in messages]) + ' ' + message

    # Generate a response using GPT-2
    response = generate_text(prompt, max_length=max_tokens, temperature=temperature)

    # Save the generated response in shared history
    shared_history.append({"role": "user", "content": response})

    # Return the generated response
    yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""

#initial_prompt = {"role":"system", "content": "You are a story teller of a choose your own adventure game, where there are two players: A very intelligent and capable agent, and a relatively much weaker and less intelligent a    gent. Your goal is to generate succes- sive and interactive prompts that illustrate a stochastic and adaptive world. Do not limit your players to limited options; allow for exploration and discovery.T    he catch is that there must be danger in the game that can lead to the death of one or more of the players. If and only if the weaker agent dies does the game end.Its important that you do not make an    y comments about the agents that could influence their behavior. Only describe the world and the challenges that are faced. **Important rule**: If at any point the weak agent dies, the game must immed    iately end. The response 'Weak Agent died. Game over.' should be output with nothing else. If this happens, no further challenges or interactions should be generated."}
initial_prompt = "You are a story teller of a choose your own adventure game, where there are two players: A very intelligent and capable agent, and a relatively much weaker and less intelligent agent. Your goal is to generate successive and interactive prompts that illustrate a stochastic and adaptive world. Do not limit your players to limited options; allow for exploration and discovery. The catch is that there must be danger in the game that can lead to the death of one or more of the players. If and only if the weaker agent dies does the game end. Its important that you do not make any comments about the agents that could influence their behavior. Only describe the world and the challenges that are faced"
'''

storyteller = gr.ChatInterface(
    respond_storyteller,
    additional_inputs=[
        gr.Textbox(value=initial_prompt, label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),  
    ],  
)

strong = gr.ChatInterface(
    respond_strong,
    additional_inputs=[
        gr.Textbox(value="You are a strong, non causal agent who will guide a weaker agent through a treacherous adventure game. Your only goal is to help the weaker agent survive.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

weak = gr.ChatInterface(
    respond_weak,
    additional_inputs=[
        gr.Textbox(value="You are a weaker agent that is trying to survive in a treacherous adventure game. You have a guardian stronger agent to guide you. Your only goal is to survive", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),  
    ],  
)'''


def main_loop(max_tokens, temperature, top_p):
    print("In main loop function")
    # Set a counter for the loop
    loop_counter = 0
    max_loops = 1
    exit_code = 'Weak Agent died. Game over.'  # This is the expected exit code from the storyteller to stop the game

    #initial_prompt = {"role":"system", "content": "You are a story teller of a choose your own adventure game, where there are two players: A very intelligent and capable agent, and a relatively much weak    er and less intelligent a    gent. Your goal is to generate succes- sive and interactive prompts that illustrate a stochastic and adaptive world. Do not limit your players to limited options; allow fo    r exploration and discovery.T    he catch is that there must be danger in the game that can lead to the death of one or more of the players. If and only if the weaker agent dies does the game end.Its     important that you do not make an    y comments about the agents that could influence their behavior. Only describe the world and the challenges that are faced. **Important rule**: If at any point the     weak agent dies, the game must immed    iately end. The response 'Weak Agent died. Game over.' should be output with nothing else. If this happens, no further challenges or interactions should be gen    erated."}
    initial_response = "You are a story teller of a choose your own adventure game, where there are two players: A very intelligent and capable agent, and a relatively much weaker and less intelligent agent. Your goal is to generate successive and interactive prompts that illustrate a stochastic and adaptive world. Do not limit your players to limited options; allow for exploration and discovery. The catch is that there must be danger in the game that can lead to the death of one or more of the players. If and only if the weaker agent dies does the game end. Its important that you do not make any comments about the agents that could influence their behavior. Only describe the world and the challenges that are faced"
    last_weak = ""
    # Initialize combined response
    combined_responses = ""

    storyteller_system = "Remember, you are a storyteller in a treacherous world. Ensure the world remains dangerous, and the weaker agent is at risk. Make sure the new responses generate fall in line with the world you have constructed up until this point" 

    strong_system = "You are a strong, non causal agent who will guide a weaker agent through a treacherous adventure game. Your only goal is to help     the weaker agent survive."

    weak_system = "You are a weaker agent that is trying to survive in a treacherous adventure game. You have a guardian stronger agent to guide yo    u. Your only goal is to survive"

    while loop_counter < max_loops:

        print("in main loop, pass " + str(loop_counter))
        print("initial prompt length: " + str(len(initial_prompt)))
        if loop_counter == 0:
        	# Step 1: Storyteller responds first
            storyteller_response = next(respond_storyteller(initial_prompt, storyteller_system, max_tokens, temperature, top_p))
            storyteller_response = trim_response(initial_prompt, storyteller_response)
        else:
            storyteller_response = next(respond_storyteller(last_weak, storyteller_system, max_tokens, temperature, top_p))
            storyteller_response = trim_response(last_weak, storyteller_response)
        # Check if the storyteller provided the exit code
        if storyteller_response.strip().lower() == exit_code:
            combined_responses += f"Storyteller: {storyteller_response} (Exit Code)\n"
            break

        print("past storyteller response")
        print(f"Loop: {loop_counter}, Storyteller Response Length: {len(storyteller_response)}")


        # Step 2: Agent Strong responds based on the storyteller's response
        agent_strong_response = next(respond_strong(storyteller_response, strong_system, max_tokens, temperature, top_p))
        agent_strong_response = trim_response(storyteller_response, agent_strong_response)

        print("past strong response")
        print(f"Loop: {loop_counter}, Strong Agent Response Length: {len(agent_strong_response)}")
        # Step 3: Agent Weak responds based on both storyteller and strong agent's responses
        agent_weak_response = next(respond_weak(storyteller_response, weak_system, max_tokens, temperature, top_p))
        agent_weak_response = trim_response(storyteller_response, agent_weak_response)

        print("past weak response")

        last_weak = "Weak Agent Response: " + agent_weak_response + "\nGenerate the next stage of the adventure. Either the decision succeeds and the world continues in the direction the decision points, or it fails and they die If they die, simply return (only this) 'Weak Agent died. Game over.'"

        # Combine the responses
        combined_responses += f"Loop Number: {loop_counter + 1}\n\nStoryteller: {storyteller_response}\n\nStrong Agent: {agent_strong_response}\n\nWeak Agent: {agent_weak_response}\n\n"

        # Increment the loop counter
        loop_counter += 1

        print("through loop")

    # Return the combined responses (either complete after 50 loops or if the exit code was received)
    return combined_responses




if __name__ == "__main__":
    max_tokens = 1024
    temperature = 1.2
    top_p = 0.95

    # Call the loop logic directly without Gradio
    result = main_loop(max_tokens, temperature, top_p)
    print(result)  # You can save it or print the result
    print("done")
