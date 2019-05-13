

def compute_validation_accuracy(model,val_loader):
    # load model to inference
    model.eval()
    
    acc_accum = 0.0
    PIXELS_IN_BATCH = PIXELS_IN_PIC * val_loader.batch_size
    for idx,(X,y) in enumerate(val_loader):            
        x = X.to(device)
        y = y.to(device)
                
        prediction = torch.argmax( model(x), dim = 1)   
        acc_accum += float(torch.sum(prediction == y))/PIXELS_IN_BATCH
        
    accuracy = acc_accum / val_loader.__len__()    
    
    return accuracy