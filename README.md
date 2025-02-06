# Improving Depth Estimation by Smoothing consecutive feames using Naive CNN

We provide all of the code needed to train our models, as well as pretrained model files for which you can run validation.

## Dataset Overview
Due to memory limitations on GitHub, we cannot upload our entire ~50k image training dataset here. We still provide data for a few scenes for you to experiment with. The full dataset can be obtained at https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2

Also due to memory limitations on GitHub, we cannot upload all of our model files saved throughout the training process for each approach. However, we have included the 3 model files that you actually need - the files for the unregularized CNN (Iteration 2), regularized CNN (Iteration 3), and transformer (Iteration 4), each at the end of training (end of epoch 9)

## Dataset Splitting
We have splitted our datasets into training set and the validation set for running the experiments. We have used NYUv2 dataset videos Contains ~400k images of indoor scenes (bedrooms, kitchens, offices, etc) and corresponding ground truth depth maps. The images are basically the consecutive frames of the video.

Note that we perform a custom train-validation split of the data, so if you try to use the small dataset of a few scenes provided here, you should run train_val_split.py so as to get a valid train-validation split on the given data before running main.py. If you run main.py without first running train_val_split.py, you will get an error like: 
#### FileNotFoundError: [Errno 2] No such file or directory: '/home/CIS6800Project/datasets/nyu_data/data/nyu2_train/bedroom_0104_out/82.png' 
because the training code is trying to retrieve an image from our original train-validation split, as opposed to one for the small dataset provided here. If you instead download the full dataset and want to evaluate our models on our original train-validation split, you should not run train_val_split.py, as the train_list.json and val_list.json files in the train_val_split directory specify the train-validation split we used for all of our experiments.

## Running the code
For running the experiment, you simply need to do: -

```bash
python main.py
```



## Parameters Tuning
You can alter the following variables within main.py according to your desired experiment:
#### model_string: set this to 'cnn', 'cnn_regularized', or 'transformer' depending on which model you want to use (it will be automatically loaded from the models folder)

#### batch_size: we set this to 10 for training our CNN-based models, and 2 for training our transformer-based model, which was performed on an NVIDIA 4090 GPU with ~25 GB of RAM - you may need to set this lower to avoid CUDA out of memory errors, depending on your hardware

batches_per_backprop: we implemented gradient accumulation - this sets the number of batches to accumulate gradients for before performing a gradient update - we set this to 1 for training our CNN-based models and 8 for training our transformer-based model, effectively making the batch size for transformer 16

train: set to True to run training, False to run validation

load_model: set to True to load saved model, False to start from scratch

epoch_to_load: epoch of training from which to load model (if loading model) - since we could only upload our models from the end of training, this must be set to 9 unless you are loading a model you trained yourself

segment_to_load: we implemented the ability to save copies of each model partway through each epoch - set this to i+1 to load the saved model from val_checkpoint_ratios[i] in epoch_to_load, set this to None to load the saved model from end of epoch_to_load - since we could only upload our models from the end of training, this must be set to None unless you are loading a model you trained yourself

val_checkpoint_ratios: set this to a list of proportions through the training at which you want to save your models and run validation, we set this to [0.25, 0.5, 0.75] in most of our experiments (no need to include 1, as validation and model saving occur automatically at the end of each epoch)

## Aside
These are some additional notebooks that we have included for reference. The iterations 0 and 1 were not directly related to this research but the idea of using an ML experiment (CNN or Transformer based approach) has its roots of improving the alignment of the maps by explicitly aligning them. The iteration 0 and iteration 1 have been defined below: -

NCUT_Visualizations.ipynb - used to generate our NCUT visualizations

### Iteraion 0


Iterations0-1.ipynb - implemented and evaluated Iterations 0 and 1

CIS6800ProjectDemo.ipynb - demo for visualizing the output depth maps produced by Iterations 2-4

Each notebook contains instructions on how to run it.

Note that the dinov2 folder has been copied from the DINOv2 GitHub repository, available at https://github.com/facebookresearch/dinov2

Also note that we initially wrote our implementations of the CNN-based adapter (model_definition_cnn.py) and of the transformer-based adapter (model_definition_transformer.py) anticipating that we might experiment with different numbers of CNN layers/transformer blocks and of past images to condition on. While these are valid directions of future work, we explored different directions we found more relevant during this project. Accordingly, we did not explore any alternate values for these parameters, and they should remain as set in our code.
