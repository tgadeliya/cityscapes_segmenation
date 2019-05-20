from torchvision import datasets
from os.path import join
import torchvision
from torch.utils.data import SubsetRandomSampler
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch

import json 

from utils import testutils
from utils import datasetutils

parser = ArgumentParser()

parser.add_argument("model")

parser.add_argument("dataset",
                    type=str,
                    )

parser.add_argument("tr_name",
                    type = str,
                    default="trained",
                    help = "Specified model from model_name/state_dict. By default trained")


args = parser.parse_args()

PATH = "../models"
PATH_TO_MODEL = join(PATH,args.name)
model = torch.load(join(PATH_TO_MODEL, args.tr_model_name))

#Validation loader
with open("../datasets/"+args.dataset) as jsonfile:
    ds_info = json.load(jsonfile)

PATH_TO_DATA = "../data/" + ds_info['data_folder']

trsf = torchvision.transforms.Compose([datasetutils.Split(), datasetutils.HorizontalFlip()])
dataset = datasetutils.CityScapes(PATH_TO_DATA, transform = trsf)

val_sampler = SubsetRandomSampler(ds_info["val_indices"])
val_loader = DataLoader(dataset, batch_size=1, sampler = val_sampler)


score = testutils.compute_validation_accuracy(model, val_loader)

