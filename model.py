import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch import Tensor
from PIL import Image
from torchvision import transforms

 
img_size = (256, 256)
trf = transforms.Compose([
        transforms.CenterCrop((512, 512)),
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]) 

reverse_trf = transforms.Compose([
    transforms.ToPILImage()
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
    
    gen_img = org_img.clone().requires_grad_(True)
    optimizer = optim.AdamW([gen_img], lr=1e-3)
    
    # pass three images to model
    gen_features = model(gen_img.to(device))
    org_features = model(org_img.to(device))
    style_features = model(style_img.to(device))
    
    for epoch in num_epoch:
        content_loss, style_loss = 0
        for gen_feature, org_feature, style_feature in zip(gen_features, org_features, style_features):
            
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
    
        if epoch+1 % 50 == 0:
            img = reverse_trf(gen_img)
            img.save(f"output_{epoch}.jpg")
            

    
def main():
    org_img_path = 'Takamura.jpg'
    style_img_path = 'style.jpg'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    alpha = 0.99
    beta = 1.0 - alpha
    num_epoch = 100
    
    org_img = load_img(org_img_path)
    style_img = load_img(style_img_path)
    
    model = VGG().to(device)
    
    train(model, org_img, style_img, alpha, beta, num_epoch, device)
    
    
if __name__ == "__main__":
    main()
    
        
        
        
            
            

    
    