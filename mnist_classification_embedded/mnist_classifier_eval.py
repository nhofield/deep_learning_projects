from src.network_classes.classes import SimpleNN
import torch
from pathlib import Path

picture_dim = 28
hidden_layer_neurons = 40

# Instantiate the model architecture
model = SimpleNN(picture_dim, hidden_layer_neurons)

# Load the model's saved state dictionary
model.load_state_dict(torch.load("models/mnist_model.pth"))

# Set the model to evaluation mode
model.eval()

data_dir = Path("./data")
cpp_code_dir = data_dir / "model_cpp_code"

for name, param in model.named_parameters():
    print(f"Parameter: {name}")
    arr = param.data
    print(arr)  # This prints the actual matrix
    print(arr.shape)  # This prints the actual matrix

    arr_shape = arr.shape
    dimension = len(arr_shape)

    if dimension > 1:
        cpp_code = "{"

        for row in range( arr_shape[0] ):

            for val_index in range( arr_shape[1] ):

                val = int(arr[row][val_index]*1000)
                cpp_code += f"{val}"
                if val_index < picture_dim**2:
                    cpp_code += f", "
            cpp_code += "\n"
        cpp_code = cpp_code[:-3]
        cpp_code += "}"

        with open( cpp_code_dir / f"{name}.txt", "w") as outfile:
            outfile.write(cpp_code)
    else:
    
        cpp_code = "{"

        for val_index in range( arr_shape[0] ):
                val = int(arr[val_index]*1000)
                cpp_code += f"{val}"
                if val_index < arr_shape[0]:
                    cpp_code += f", "
        
        cpp_code += "}"

        with open( cpp_code_dir / f"{name}.txt", "w") as outfile:
            outfile.write(cpp_code)
