import torch

print(f"¿CUDA está disponible?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cantidad de GPUs detectadas: {torch.cuda.device_count()}")
else:
    print("PyTorch NO está detectando tu GPU. Se está ejecutando en la CPU.")