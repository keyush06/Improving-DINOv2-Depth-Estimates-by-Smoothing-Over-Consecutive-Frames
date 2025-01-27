from utils import *
from train_and_val import *
from model_definition_cnn import CNNModel
from model_definition_transformer import *

model_string = 'cnn' # in 'cnn', 'cnn_regularized', 'transformer'
num_heads = 77 # is only used if model_string is 'transformer'

batch_size = 10 # set to 2 if model_string is 'transformer', else 10
batches_per_backprop = 1 # set to 8 if model_string is 'transformer', else 1

train = False # set to True to run training, False to run validation
load_model = False # set to True to load saved model, False to start from scratch

epoch_to_load = 9 # epoch of training from which to load model (if loading model)
segment_to_load = None # set this to i+1 to load model from val_checkpoint_ratios[i] through selected epoch, set to None to load model from end of selected epoch

val_checkpoint_ratios = [0.25, 0.5, 0.75] # points in each epoch to save model and run validation

backbone_size = "small" # in ("small", "base", "large", or "giant")
head_dataset = "nyu" # in ("nyu", "kitti")
head_type = "dpt" # in ("linear", "linear4", "dpt")

data_directory = './datasets/nyu_data/data/nyu2_train'

filename_train = './train_val_split/train_list.json'
filename_val = './train_val_split/val_list.json'

image_height = 480
image_width = 640

feature_height = 35
feature_width = 46

num_epochs = 10
learning_rate = 1e-5

backbone_name, backbone_model = load_backbone(backbone_size=backbone_size)

model = load_dino_model(backbone_name, backbone_model, backbone_size=backbone_size, head_dataset=head_dataset, head_type=head_type)

with open(filename_train, 'r') as f:
    train_list = json.load(f)

with open(filename_val, 'r') as f:
    val_list = json.load(f)

train_list = [sample for sample in train_list if sample[1] > 1]
val_list = [sample for sample in val_list if sample[1] > 1]

num_images_train = len(train_list)
num_images_val = len(val_list)

transform = make_depth_transform()

if train:
    print("Training " + model_string)
if not train:
    print("Running Validation for " + model_string)
    load_model = True

if 'cnn' in model_string:
    adapter_model = CNNModel(1, 1)
else:
    adapter_model = TransformerModel(1, 1, num_heads, 1)
adapter_model.cuda()
optimizer = torch.optim.Adam(adapter_model.parameters(), lr=learning_rate)

num_batches_train = int(num_images_train/batch_size)
val_checkpoints = (num_batches_train * np.array(val_checkpoint_ratios)).astype(int)

if load_model:
    if segment_to_load is None:
        adapter_model.load_state_dict(torch.load('./models/' + model_string + '_model_epoch_' + str(epoch_to_load) + '.pth', weights_only=True))
        start_epoch = epoch_to_load + 1
        start_batch_index = 0
        val_flag = 0
    else:
        adapter_model.load_state_dict(torch.load('./models/' + model_string + '_model_epoch_' + str(epoch_to_load) + '_segment_' + str(segment_to_load) + '.pth', weights_only=True))
        start_epoch = epoch_to_load
        if val_checkpoints[segment_to_load - 1] % batches_per_backprop == batches_per_backprop - 1:
            start_batch_index = val_checkpoints[segment_to_load - 1] + 1
        else:
            start_batch_index = val_checkpoints[segment_to_load - 1] - (val_checkpoints[segment_to_load - 1] % batches_per_backprop)
        val_flag = segment_to_load
else:
    start_epoch = 0
    start_batch_index = 0
    val_flag = 0

original_mse_train, updated_mse_train, original_mse_val, updated_mse_val = load_training_logs(num_epochs, len(val_checkpoints), model_string)

difference_train = torch.zeros(num_epochs)
difference_val = torch.zeros(num_epochs, len(val_checkpoints) + 1)

if train:

    for epoch in range(start_epoch, num_epochs):

        if epoch == start_epoch:
            original_mse_current_epoch_train_mean, updated_mse_current_epoch_train_mean, original_mse_val, updated_mse_val = train_loop(original_mse_val, updated_mse_val, train_list, val_list, data_directory, transform, model, adapter_model, optimizer, image_height, image_width, feature_height, feature_width, num_images_train, num_images_val, batch_size, batches_per_backprop, epoch, val_checkpoints, val_flag, model_string, start_batch_index=start_batch_index)
        else:
            original_mse_current_epoch_train_mean, updated_mse_current_epoch_train_mean, original_mse_val, updated_mse_val = train_loop(original_mse_val, updated_mse_val, train_list, val_list, data_directory, transform, model, adapter_model, optimizer, image_height, image_width, feature_height, feature_width, num_images_train, num_images_val, batch_size, batches_per_backprop, epoch, val_checkpoints, val_flag, model_string, start_batch_index=0)

        original_mse_train[epoch] = original_mse_current_epoch_train_mean
        updated_mse_train[epoch] = updated_mse_current_epoch_train_mean

        torch.save(adapter_model.state_dict(), './models/' + model_string + '_model_epoch_' + str(epoch) + '.pth')

        torch.save(original_mse_train, './training_logs/original_mse_train_' + model_string + '.pt')
        torch.save(updated_mse_train, './training_logs/updated_mse_train_' + model_string + '.pt')

        original_mse_current_epoch_val_mean, updated_mse_current_epoch_val_mean = val_loop(val_list, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, num_images_val, batch_size, epoch, model_string, end_of_epoch=True)
        original_mse_val[epoch, -1] = original_mse_current_epoch_val_mean
        updated_mse_val[epoch, -1] = updated_mse_current_epoch_val_mean

        torch.save(original_mse_val, './training_logs/original_mse_val_' + model_string + '.pt')
        torch.save(updated_mse_val, './training_logs/updated_mse_val_' + model_string + '.pt')

        val_flag = 0

else:
    if segment_to_load is None:
        original_mse_current_epoch_val_mean, updated_mse_current_epoch_val_mean = val_loop(val_list, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, num_images_val, batch_size, epoch_to_load, model_string, end_of_epoch=True)
        original_mse_val[epoch_to_load, -1] = original_mse_current_epoch_val_mean
        updated_mse_val[epoch_to_load, -1] = updated_mse_current_epoch_val_mean
    else:
        original_mse_current_epoch_val_mean, updated_mse_current_epoch_val_mean = val_loop(val_list, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, num_images_val, batch_size, epoch_to_load, model_string, end_of_epoch=False, training_percentage=100*segment_to_load/(len(val_checkpoints) + 1))
        original_mse_val[epoch_to_load, segment_to_load - 1] = original_mse_current_epoch_val_mean
        updated_mse_val[epoch_to_load, segment_to_load - 1] = updated_mse_current_epoch_val_mean

    torch.save(original_mse_val, './training_logs/original_mse_val_' + model_string + '.pt')
    torch.save(updated_mse_val, './training_logs/updated_mse_val_' + model_string + '.pt')
