import paddle
from paddle import nn
from ..base.theseus_layer import TheseusLayer


class Encoder(TheseusLayer):
    def __init__(self, num_channels, num_filters):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2D(in_channels=num_channels,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn1   = nn.BatchNorm(num_filters,act="relu")
        
        self.conv2 = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn2   = nn.BatchNorm(num_filters,act="relu")
        
        self.pool  = nn.MaxPool2D(kernel_size=2,stride=2,padding="SAME")
        
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_conv = x           
        x_pool = self.pool(x)
        return x_conv, x_pool
    
    
class Decoder(TheseusLayer):
    def __init__(self, num_channels, num_filters):
        super(Decoder,self).__init__()
        self.up = nn.Conv2DTranspose(in_channels=num_channels,
                                    out_channels=num_filters,
                                    kernel_size=2,
                                    stride=2,
                                    padding=0)

        self.conv1 = nn.Conv2D(in_channels=num_filters*2,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn1   = nn.BatchNorm(num_filters,act="relu")
        
        self.conv2 = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn2   = nn.BatchNorm(num_filters,act="relu")
        
    def forward(self,input_conv,input_pool):
        x = self.up(input_pool)
        h_diff = (input_conv.shape[2]-x.shape[2])
        w_diff = (input_conv.shape[3]-x.shape[3])
        pad = nn.Pad2D(padding=[h_diff//2, h_diff-h_diff//2, w_diff//2, w_diff-w_diff//2])
        x = pad(x)                               
        x = paddle.concat(x=[input_conv,x],axis=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x
    
class U_Net(TheseusLayer):
    def __init__(self,num_classes=3):
        super(U_Net,self).__init__()
        self.down1 = Encoder(num_channels=3, num_filters=64) 
        self.down2 = Encoder(num_channels=64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)
        
        self.mid_conv1 = nn.Conv2D(512,1024,1)               
        self.mid_bn1   = nn.BatchNorm(1024,act="relu")
        self.mid_conv2 = nn.Conv2D(1024,1024,1)
        self.mid_bn2   = nn.BatchNorm(1024,act="relu")

        self.up4 = Decoder(1024,512)
        self.up3 = Decoder(512,256)
        self.up2 = Decoder(256,128)
        self.up1 = Decoder(128,64)
        
        self.last_conv = nn.Conv2D(64,num_classes,1)
        
    def forward(self,inputs):
        x1, x = self.down1(inputs)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)
        
        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        
        x = self.last_conv(x)
        
        return x


def UNet(num_classes=3, **kwargs):
    return U_Net(num_classes, **kwargs)