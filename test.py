import torch

print("CUDA Version: ", torch.version.cuda)
print("Is CUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Device Count: ", torch.cuda.device_count())
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
    print("CUDA Device Properties: ", torch.cuda.get_device_properties(0))
else:
    print("CUDA is not available. Check your installation.")
