__Description__
    
# Image segmentation 
  Implementation of UNet for image segmenatation on cityscape dataset.

## Dataset



## Simplest baseline
  Because I am using accuracy metric, it is easy to present the simplest baseline based on constant output. As we can see on histogram below, class 4 the most
  Baseline = 4th_class_pixels / total_pixels =

## Usage
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
         
## Models directory   
## Improvements
   
