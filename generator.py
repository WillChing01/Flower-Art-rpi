import torch,torchvision
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image

stats=(0.5,0.5,0.5),(0.5,0.5,0.5)

def denorm(img_tensors):
    return img_tensors*stats[1][0]+stats[0][0]

if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

latent_size=100

generator=nn.Sequential(
    #in latent_sizex1x1

    nn.ConvTranspose2d(latent_size,1024,kernel_size=4,stride=1,padding=0,bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(True),
    #out 1024x4x4

    nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    #out 512x8x8

    nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    #out 256x16x16

    nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    #out 128x32x32

    nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    #out 64x64x64

    nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1,bias=False),
    nn.Tanh()
    #out 3x128x128
)

generator=to_device(generator,device)

try:
    generator.load_state_dict(torch.load('G.ckpt'))
##    print("Successfully loaded model!")
except: pass
##    print("Error! Could not load pre-trained model!")

##import matplotlib.pyplot as plt
##
##seed=torch.randn(16,latent_size,1,1,device=device)
##
##img=generator(seed)
##fig,ax=plt.subplots(figsize=(4,4))
##ax.set_xticks([]);ax.set_yticks([])
##ax.imshow(make_grid(denorm(img).cpu().detach(),nrow=4).permute(1,2,0))
##
##plt.show()
