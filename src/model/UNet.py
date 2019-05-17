import torch.nn as nn
import torch

def make_block(in_channels, out_channels, dev_type):
    block = nn.Sequential(nn.ReplicationPad2d(1),
                              nn.Conv2d(in_channels, out_channels, 3),
                              nn.ReLU(),
                              nn.BatchNorm2d(out_channels),
                              nn.ReplicationPad2d(1),
                              nn.Conv2d(out_channels, out_channels, 3),
                              nn.ReLU(),
                              nn.BatchNorm2d(out_channels)
                             )
    if (dev_type == "cuda"):
        block.type(torch.cuda.FloatTensor)
    return block

class UNet(nn.Module):
    def __init__(self, basic_chl, dev_type = "cuda", num_classes = 30, image_channels = 3):
        super(UNet, self).__init__()
        
        #Calculate channels based on basic channel
        BL1_chl, BL2_chl, BL3_chl, BL4_chl, BL5_chl= [basic_chl * i for i in [1,2,4,8,16]]
        
        #Encoder blocks
        self.ENC_BL1 = make_block(image_channels,BL1_chl, dev_type)
        self.ENC_BL2 = make_block(BL1_chl, BL2_chl, dev_type)
        self.ENC_BL3 = make_block(BL2_chl, BL3_chl, dev_type)
        self.ENC_BL4 = make_block(BL3_chl, BL4_chl, dev_type)
        self.ENC_BL5 = make_block(BL4_chl, BL5_chl, dev_type)
        
        #MaxPool for downsampling
        self.MaxPool = nn.MaxPool2d(2,2)
        
        #UpConv for upsampling
        self.UpConv1 = nn.ConvTranspose2d(BL5_chl, BL4_chl, 2, stride = 2)
        self.UpConv2 = nn.ConvTranspose2d(BL4_chl, BL3_chl, 2, stride = 2)
        self.UpConv3 = nn.ConvTranspose2d(BL3_chl, BL2_chl, 2, stride = 2)
        self.UpConv4 = nn.ConvTranspose2d(BL2_chl, BL1_chl, 2, stride = 2)
        
        #Decoder blocks
        self.DEC_BL1 = make_block(BL5_chl, BL4_chl, dev_type)
        self.DEC_BL2 = make_block(BL4_chl, BL3_chl, dev_type)
        self.DEC_BL3 = make_block(BL3_chl, BL2_chl, dev_type)
        self.DEC_BL4 = make_block(BL2_chl, BL1_chl, dev_type)
        
        # Last convolution
        self.Final_CONV = nn.Conv2d(basic_chl, num_classes, 1)
        
    
    def forward(self,X):
        # Encoder
        ENC_1_out = self.ENC_BL1.forward(X)
        MaxENC_1_out = self.MaxPool.forward(ENC_1_out)
        
        ENC_2_out = self.ENC_BL2.forward(MaxENC_1_out)
        MaxENC_2_out = self.MaxPool.forward(ENC_2_out)
        
        ENC_3_out = self.ENC_BL3.forward(MaxENC_2_out)
        MaxENC_3_out = self.MaxPool.forward(ENC_3_out)
        
        ENC_4_out = self.ENC_BL4.forward(MaxENC_3_out)
        MaxENC_4_out = self.MaxPool.forward(ENC_4_out)
                
        # Bridge to Decoder
        ENC_5_out = self.ENC_BL5.forward(MaxENC_4_out)
        UpENC_5_out = self.UpConv1.forward(ENC_5_out)  
        
        #Decoder
        DEC_1_in = torch.cat((UpENC_5_out,ENC_4_out), dim = 1)
        DEC_1_out = self.DEC_BL1.forward( DEC_1_in)
        UpDEC_1_out = self.UpConv2.forward(DEC_1_out)
        
        
        DEC_2_in = torch.cat((UpDEC_1_out,ENC_3_out), dim = 1)
        DEC_2_out = self.DEC_BL2.forward(DEC_2_in)
        UpDEC_2_out = self.UpConv3.forward(DEC_2_out)
        
        DEC_3_in = torch.cat((UpDEC_2_out,ENC_2_out), dim = 1)
        DEC_3_out = self.DEC_BL3.forward(DEC_3_in)
        UpDEC_3_out = self.UpConv4.forward(DEC_3_out)
        
        DEC_4_in = torch.cat((UpDEC_3_out,ENC_1_out), dim = 1)
        DEC_4_out = self.DEC_BL4.forward(DEC_4_in)
          
        #Final Convolution    
        segmenatation_map = self.Final_CONV.forward(DEC_4_out)
    
        return segmenatation_map
