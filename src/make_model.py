import argparse
from os import mkdir

import datetime
import json

import torch
import model.UNet as Net



parser = argparse.ArgumentParser(description='Make UNet model with fixed structure and specified number of channels in the convolution layers.\
    Create folder ../models/model_name  model parameters, meta information and inside and meta info')

parser.add_argument('-Nchl', 
                    type=int,
                    help="Base for channels in two sides of the model.\
                         Create N, N*2, N*4, N*8, N*16, N*8, N*4, N*2, N channels in every block of U-shaped model Net. In the original paper was set as 64.")

parser.add_argument('-name', 
                    type=str,
                    help="Name of created model. Directory with this name will be created in ../models")

parser.add_argument('--device',
                    type=str,
                    default = "cuda", 
                    help='Put model on specified device. By default CUDA device is specified.')

args = parser.parse_args()

PATH = "../models/"

path_to_model = PATH+args.name
mkdir(path_to_model)


# Save model
path_to_param = path_to_model+"/state_dict"
mkdir(path_to_param)
model = Net.UNet(args.Nchl, dev_type = args.device)
torch.save(model.state_dict(), path_to_param+"/init.pth")


# Create meta json file with basic info
meta = open(path_to_model+"/meta.json","a+")
meta_dict = { "date" : datetime.datetime.now().strftime("%Y-%m-%d"),
              "time": datetime.datetime.now().strftime("%H:%M:%S"),
              "device" : args.device,
              "model_name" : args.name,
              "Nchl": args.Nchl
}
meta.write(json.dumps(meta_dict))
meta.close()

# Put model specification into txt file

f = open("model_architecture.txt", "a+")
f.write(str(model))