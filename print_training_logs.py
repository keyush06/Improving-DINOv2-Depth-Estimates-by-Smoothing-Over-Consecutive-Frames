from utils import *

model_string = 'cnn' # in 'cnn', 'cnn_regularized', 'transformer'
# Setting of next two values does not matter
num_epochs = 10
num_val_checkpoints = 3

original_mse_train, updated_mse_train, original_mse_val, updated_mse_val = load_training_logs(num_layers, num_past_images, num_epochs, num_val_checkpoints, model_string)

difference_train = updated_mse_train - original_mse_train
difference_val = updated_mse_val - original_mse_val

print("Original MSE Train for " + model_string + " = " + str(original_mse_train))
print("Updated MSE Train for " + model_string + " = " + str(updated_mse_train))
print("Difference Train for " + model_string + " = " + str(difference_train))
print("\nOriginal MSE Val for " + model_string + " = " + str(original_mse_val))
print("Updated MSE Val for " + model_string + " = " + str(updated_mse_val))
print("Difference Val for " + model_string + " = " + str(difference_val))

