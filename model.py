import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch import Tensor
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image
 
img_size = (256, 256)
trf = transforms.Compose([
        transforms.CenterCrop((512, 512)),
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]) 


def load_img(path):
    img = trf(Image.open(path))
    return img
    

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19(pretrained=True)
        self.chosen_features = [0, 5, 10, 19]
    
    def forward(self, x):
        return_features = [] 
        for num, feature in enumerate(self.model.features):
            x = feature(x)
            if num in self.chosen_features:
                return_features.append(x)
        
        return return_features
            
def train(model: nn.Module, org_img: Tensor, style_img: Tensor, alpha=0.99, beta=0.01, num_epoch=100, device='cuda'):
    
    gen_img = torch.rand_like(org_img).to(device).requires_grad_(True)
    optimizer = optim.Adam([gen_img], lr=1e-1)

    for epoch in tqdm(range(num_epoch)):
        # print(epoch)
        gen_features = model(gen_img)
        org_features = model(org_img.to(device))
        style_features = model(style_img.to(device))
        
        content_loss = style_loss = 0
        for gen_feature, org_feature, style_feature in zip(gen_features,
                                                           org_features,
                                                           style_features):
            
            # calc content loss by mse
            content_loss += torch.mean((org_feature - gen_feature)**2)
            
            # Gram matrix for style loss
            C, H, W = gen_feature.size()
            gen_2d = gen_feature.view(C, H*W)
            style_2d = style_feature.view(C, H*W)
            
            # calc gram matrix
            gen_gram = gen_2d @ gen_2d.T
            style_gram = style_2d @ style_2d.T
            
            # style loss
            style_loss += torch.mean((gen_gram - style_gram)**2)
    
        # total loss
        loss = alpha*content_loss + beta*style_loss
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(epoch + 1 % 10 == 0)
        if (epoch+1) % 100 == 0:
            print('saving img...')
            save_image(gen_img.detach().cpu(), f"outputs/output3_{epoch}.jpg")
            

    
def main():
    org_img_path = 'Takamura.jpg'
    style_img_path = 'style2.jpeg'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    alpha = 2.0
    beta = 1e-5
    num_epoch = 5000
    
    org_img = load_img(org_img_path)
    style_img = load_img(style_img_path)
    
    model = VGG().to(device)
    model.eval()
    
    train(model, org_img, style_img, alpha, beta, num_epoch, device)
    
    
if __name__ == "__main__":
    main()
    
        
        
        
            
            

    
    