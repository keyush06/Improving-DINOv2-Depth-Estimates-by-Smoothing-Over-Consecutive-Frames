from utils import *

data_directory = './datasets/nyu_data/data/nyu2_train'

datasets, num_images_dataset, num_images = enumerate_datasets(data_directory)

split_ratio = 0.8

filename_train = './train_val_split/train_list.json'
filename_val = './train_val_split/val_list.json'

train_list = []
val_list = []

num_images_train = 0
num_images_val = 0

for i in range(len(datasets)):
    for j in range(1, num_images_dataset[i]):
        train_choice = np.random.rand(1) < split_ratio
        if train_choice:
            train_list.append([datasets[i], j])
            num_images_train += 1
        else:
            val_list.append([datasets[i], j])
            num_images_val += 1

print("Num Images = " + str(num_images))
print("Num Images Train = " + str(num_images_train))
print("Num Images Val = " + str(num_images_val))

with open(filename_train, 'w') as f:
    json.dump(train_list, f)

with open(filename_val, 'w') as f:
    json.dump(val_list, f)
