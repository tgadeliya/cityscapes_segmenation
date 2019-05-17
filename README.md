__Project Description__
  Implementation of UNet for image segmenatation on cityscape dataset.\n
  Link:https://www.cityscapes-dataset.com/

## Dataset


## Usage
    Note:All commands use fixed folders to load data,datasets,models etc. For that reason you only need to put names, not pathes.
  0. Create dataset file that stores information about train/val indices, batch size and  name of the data folder in data/ directory.
        ```python 
        python3 make_dataset.py -data_folder cityscapes -name dataset_name -val_percent 0.2 -tr_batch_size 8
        ```
  1. To create model run script src/make_model.py
        ```python 
        python3 make_model.py -Nchl 64 -name your_model_name --device cuda
        ```
     This step will create directory in models/your_model_name with parameters and META-info inside.
     See Models directory for more information.
  2. Start training created model:
       ```python
       python3 train.py -name your_model_name -dataset path_to_data -hp mode
       ```
     Train model with specified hyperparameters or turn on hparameter search mode(randoms search, grid search).
  3. Evaluate model
      ```python
       python3 eval.py -model your_model_name  -dataset dataset_name   
       ```
         
## Models directory   


## Simplest baseline
 ("images/histogram.png")
 pictures of classes
 Because I am using accuracy metric, it is easy to present the simplest baseline based on constant output. As we can see on histogram below, class 4 the most
  Baseline = 4th_class_pixels / total_pixels =

## Improvements
