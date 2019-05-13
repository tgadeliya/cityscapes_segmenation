from argparse import ArgumentParser
import numpy as np
import os 
import json

parser = ArgumentParser()

parser.add_argument("data_folder",
                    type=str,
                    default="cityscapes",
                    help = "Name of the folder with images in data/"
                    )

parser.add_argument("name",
                    type=str,
                    default="Cityscapes",
                    help = "Name of JSON file with created dataset in datasets/"
                    )


parser.add_argument("val_percent",
                    type=float,
                    default = 0.2,
                    help = "Fraction of images used to calculate validation score."
                    )

parser.add_argument("tr_batch_size",
                    type=int,
                    default=8,
                    help = "Size of the batch that will be used during the training."
                    )

args = parser.parse_args()


PATH = "../data/"
PATH_TO_DATA = os.path.join(PATH, args.data_folder)

dataset_length = len(os.listdir(PATH_TO_DATA))-1
indices = list(range(dataset_length))
np.random.shuffle(indices)

idx = int(round(args.val_percent * dataset_length))
train_indices,val_indices = indices[idx:], indices[:idx]

dict_info = {
    "data_folder": args.data_folder,
    "batch_size" : args.tr_batch_size,
    "train_indices": train_indices,
    "val_indices": val_indices, 
}


dataset = open("../datasets/{}.json".format(args.name), "a+")
dataset.write(json.dumps(dict_info))
