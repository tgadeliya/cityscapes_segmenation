from torchvision import datasets
from os.path import join
from torch.utils.data import SubsetRandomSampler
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch

from utils import testutils


parser = ArgumentParser()

parser.add_argument("name")
parser.add_argument("tr_model_name",
                    type = str,
                    default="trained")
parser.add_argument("dataset",
                    type=str,
                    default=None)

args = parser.parse_args()

PATH = "../models"
PATH_TO_MODEL = join(PATH,args.name)


model = torch.load(join(PATH_TO_MODEL, args.tr_model_name))

#Validation loader
indices = open(PATH_TO_MODEL+"val_indices.txt", 'r')
ind_string = indices.readline(1) 
val_ind = [int(x) for x in ind_string.split(',')]
val_sampler = SubsetRandomSampler(val_ind)
val_loader = DataLoader()


score = testutils.compute_validation_accuracy(model, val_loader)

