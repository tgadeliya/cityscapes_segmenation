import torch
import numpy as np

class CityScapes(torch.utils.data.Dataset):
    def __init__(self, path, transform = None):
        self.data_folder = path
        self.len = len(os.listdir(path))
        self.transform = transform
        
    def __getitem__(self, idx):
        path_to_file = os.path.join(self.data_folder,os.listdir(self.data_folder)[idx-1])
        image = io.imread(path_to_file)
        
        if self.transform:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return self.len





def RGB_to_idx(image, arr_to_idx):
    image = image.dot(np.array([65536, 256, 1], dtype='int32'))
    result = np.ndarray(shape=image.shape, dtype=int)
    result[:,:] = -1
    for rgb, idx in arr_to_idx.items():
        rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        result[image==rgb] = idx
    return result

def idx_to_RGB(image, idx_to_rgb):
    result = np.ndarray(shape=(256,256,3), dtype=int)
    result[:,:,:] = -1
    for x in range(256):
      for y in range(256):
        result[x][y] = idx_to_arr[int(image[x][y])]
    return result
  
  
  
check_array = np.array(
    [[116, 17, 36],[152, 43,150],
     [106,141, 34],[ 69, 69, 69],
     [  2,  1,  3],[127, 63,126],
     [222, 52,211],[  2,  1,140],
     [ 93,117,119],
     [180,228,182],[213,202, 43],
     [ 79,  2, 80],[188,151,155],
     [  9,  5, 91],[106, 75, 13],
     [215, 20, 53],[110,134, 62],
     [  8, 68, 98],[244,171,170],
     [171, 43, 74],[104, 96,155],
     [ 72,130,177],[242, 35,231],
     [147,149,149],[ 35, 25, 34],
     [155,247,151],[ 85, 68, 99],
     [ 71, 81, 43],[195, 64,182],
     [146,133, 92]]
)

arr_to_idx = {tuple(arr):idx for idx,arr in enumerate(check_array)}
idx_to_arr = {idx: arr for idx,arr in enumerate(check_array)}



# class Split_N(object):
#     def __call__(self,sample):
#       return sample[:,:256,:].astype("float32"), sample[:,256:,:].astype("float32")

class Split(object):
    def __call__(self,sample):
      return sample[:,:256,:], sample[:,256:,:]





# class HorizontalFlip_Normalize(object):
    
#     def __init__(self,mean,std):
#       self.mean = mean
#       self.std = std
      
#     def __call__(self,sample):
#         X,y = sample
        
#         y = RGB_to_idx(y,arr_to_idx)
#         if np.random.random() > 0.5:
#             X[:], y[:] = X[:,::-1,:],y[:,::-1]
            
#         X = transforms.functional.to_tensor(X)
#         return (F_normalize(X, self.mean, self.std), torch.from_numpy(y))
      
class HorizontalFlip(object):
   
    def __call__(self,sample):
        X,y = sample
        
        y = RGB_to_idx(y,arr_to_idx)
        if np.random.random() > 0.5:
            X[:], y[:] = X[:,::-1,:],y[:,::-1]
            
        X = transforms.functional.to_tensor(X)
        return (X, torch.from_numpy(y))
      
            



