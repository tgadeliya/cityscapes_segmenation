
def make_sample(params):
    sample_dict={}
  
    for key,value in params.items():
        if  (key == "optimizer"):
            sampled_param = np.random.choice(value)  
        elif (key == "loss"):
            sampled_param = torch.nn.CrossEntropyLoss
    #     elif (key == "scheduler"):
    #       sch  = np.random.choice(value)
    #       sampl_param = [np.random.choice(x) for x in sch[1:]]
    #       sampled_param = [sch[0],sampl_param]
        elif (key == "lr_aneal_factor" or key == "lr_aneal_factor") :
            sampled_param = np.random.choice(value)
        else:
            sampled_param = 10 ** np.random.uniform(low = value[0], high = value[1]) 
        sample_dict[key] = sampled_param
        
    return sample_dict

def Random_search(chan_fact, dict_of_params, N, train_loader, val_loader):
    dict_evaluations = {}
    
    for i in range(N):
        
        # Get random parameters
        sample = make_sample(dict_of_params)
    
        # Define model
        model_def = UNet(chan_fact)
        loss = sample["loss"](weight)
        
        if (device.type == "cuda"):
            model_def.type(torch.cuda.FloatTensor)
            loss.type(torch.cuda.FloatTensor)
        
        if (sample["optimizer"] == "Adam"):
            optimizer = torch.optim.Adam(model_def.parameters(), lr = sample["lr"], weight_decay = sample["L2"])
        elif (sample["optimizer"] == "SGD"):
            optimizer = torch.optim.SGD(model_def.parameters(), lr = sample["lr"], momentum = 0.9 , weight_decay = sample["L2"] )         
        
#         if (sample["scheduler"][0] == "Cyclic"):
#             scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, *sample["scheduler"][1])
#         elif (sample["scheduler"][0] == "Cosine"):
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, *sample["scheduler"][1])
#         elif (sample["scheduler"][0] == "ReduceLRonPlateau"):
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (optimizer, *sample["scheduler"][1])
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size = sample["lr_anneal_size"], gamma = sample["lr_anneal_factor"] )
                       
        #perform training
        train_loss_h, train_acc, val_acc_history = train_model(model_def, optimizer, loss, scheduler, train_loader, val_loader = val_loader,verbose = False, num_epochs = 20)
        
        
        #dict_evaluations.append([sample, [train_loss_h, train_acc, val_acc_history]])
                  
        print("Sampled {0}".format(i))    
        dict_evaluations[tuple([(name,val) for name,val in sample.items()])] = [train_loss_h, train_acc, val_acc_history]
        
    return dict_evaluations