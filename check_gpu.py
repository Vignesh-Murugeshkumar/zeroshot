import torch

print("=" * 60)
print("GPU/CUDA Detection Report")
print("=" * 60)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU detected - running on CPU")
print("=" * 60)
