import paddle

class PSNRLoss(paddle.nn.Layer):
   def __init__(self):
       super(PSNRLoss, self).__init__()

   def forward(self, input, label):
       return 100 - 20 * paddle.log10(((input - label)**2).mean(axis = [1,2,3])**-0.5)