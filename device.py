import torch

def list_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
    else:
        print("No GPUs available.")

if __name__ == "__main__":
    list_gpus()

