import torch
import datetime
 
from utils.testutils import compute_validation_accuracy

def make_histogram(dataset):
  dict_classes  = { idx:0 for idx in range(30)}
  
  for i in range(dataset.len):
      _, y = dataset[i]
      for class_idx in dict_classes.keys():
        dict_classes[class_idx] += np.sum(np.where(y == class_idx, 1, 0))
  return dict_classes 

def make_weight(dict_hist):
    num_class = len(dict_hist.keys())
    weight = np.empty(num_class)
    
    total_pixels = np.sum( list(dict_hist.values()) )
    
    for i in range(num_class):
      weight[i] =  dict_hist[i] /total_pixels
    return torch.Tensor(weight)



def train_model(log, model_name ,model, optimizer, loss_function, scheduler_obj ,train_loader, val_loader = None, num_epochs=5):         
    train_loss_history, train_acc_history, val_acc_history = [], [], []
    log.write(datetime.datetime.now().strftime("%m-%d %H:%M"))
    
    scheduler = scheduler_obj

    for epoch in range(num_epochs):
        model.train()
    
        train_acc_accum = 0.0
        running_loss = 0
        
        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x) # batch_size x num_classes x height x width
            pred = torch.argmax(output, dim=1)
            
            optimizer.zero_grad()
            
            loss = loss_function(output,y)
            train_acc_accum += float(torch.sum(pred == y))/ (x.shape[0] * 256 * 256) ##ACC_METRIC
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_acc = train_acc_accum / train_loader.__len__()
        loss_norm = running_loss / train_loader.__len__()
        
        if ( (epoch > 1) and ((loss_norm > 10) or train_acc < 0.001)):
            return [],[],[]
        
        val_acc = "-" if (val_loader == None) else compute_validation_accuracy(model, val_loader)
        scheduler.step(loss_norm)
    
        log.write( "Epoch: {0}, tr_loss = {1:.4f} , tr_acc = {2:.3f}, val_acc = {3}".format(epoch+1, loss_norm, train_acc, val_acc) )
        
        train_loss_history.append(loss_norm)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
    return train_loss_history, train_acc_history, val_acc_history