from argparse import ArgumentParser
import numpy as np
import torch 
import torch as nn
import torchvision 
import json
import os
from torch.utils.data import DataLoader


from utils import datasetutils
from utils import trainutils
from utils import RandomSearch
#from utils import GridSearch


# parse arguments
parser = ArgumentParser(description = "Train model on given dataset. You can resume previous training or start new.")

parser.add_argument('model',
                    type = str,
                    help = "Name of the model in /models/")
parser.add_argument('dataset',
                    type = str,
                    help = "Name of the dataset in /datasets/")
parser.add_argument('epochs',
                    type = int,
                    help = "Number of epochs to train.")

parser.add_argument('optim',
                    type = str,
                    default = "Adam",
                    help = "SGD or Adam. Default - Adam")

parser.add_argument('train_name',
                    type = str,
                    default = "train",
                    help = "Name to store training progress and logs")


# parser.add_argument('hpsearch')
# parser.add_argument('resume',
#                     type=bool,
#                     default=False,
#                     )

args = parser.parse_args()


## PARSE JSON DATASET
with open("../datasets/" + args.dataset) as jsonfile:    
    ds_info = json.load(jsonfile)

PATH_TO_DATA = "../data/" + ds_info['data_folder']

###Creating Dataset
## Default transformations
trsf = trsf = torchvision.transforms.Compose([datasetutils.Split(), datasetutils.HorizontalFlip()])
dataset = datasetutils.CityScapes(PATH_TO_DATA, transform = trsf)

train_indices = ds_info["train_indices"]
val_indices = ds_info["val_indices"]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size = ds_info['batch_size'], sampler = train_sampler )
val_loader = DataLoader(dataset, batch_size = ds_info['batch_size'], sampler= val_sampler)


# load model
path_to_model = "../models/"+ args.model

path_to_model = path_to_model + "/state_dict/" + ("trained.pth" if (args.resume) else "init.pth")
model = torch.load(path_to_model)

# load meta info about device
device = torch.device()

# create optimizer, scheduler, loss and train model

if (args.optim == "adam") :
    optim = torch.optim.Adam(model.parameters(), lr = 1 , weight_decay =  2)
else:
    optim = torch.optim.SGD(model.parameters(), lr = 1  , momentum = 0.9) 

loss = nn.CrossEntropy()

if (args.scheduler == "step"):
    scheduler = torch.optim.lr_scheduler.StepLR(optim) 
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optim)


train_log = open(os.path.join(path_to_model,args.train_name ), "a+")

trainutils.train_model(train_log, path_to_model ,model, optim,loss, scheduler , train_loader,scheduler  ,num_epochs=args.epochs)


#save model to models/state_dict/trained.pth
torch.save({
            'epoch': args.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'lr_scheduler' : scheduler,
            }, PATH = path_to_model+"/state_dict/"+ args.train_name
)
