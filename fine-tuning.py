import os
import torch
import glob
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from ESRGAN.model import RealESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from loss_function import CombinedLoss
from torchvision.models import vgg19
from pytorch_pretrained_vit import ViT

class ImageDataset(Dataset):
    def __init__(self, hr_folder, lr_folder):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.hr_images = sorted(glob.glob(f"{hr_folder}/*.*"))
        self.lr_images = sorted(glob.glob(f"{lr_folder}/*.*"))
        self.resize_hr = Resize((256, 256))  # Change the target size as needed
        self.resize_lr = Resize((64, 64))    # Change the target size as needed
        self.to_tensor = ToTensor()

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_images[idx]).convert("RGB")
        lr_image = Image.open(self.lr_images[idx]).convert("RGB")
        
        hr_image = self.resize_hr(hr_image)
        lr_image = self.resize_lr(lr_image)
        
        hr_image = self.to_tensor(hr_image)
        lr_image = self.to_tensor(lr_image)

        return {"hr": hr_image, "lr": lr_image}

    def __len__(self):
        return len(self.hr_images)

hr_folder = "Dataset/HighResolution"
lr_folder = "Dataset/LowResolution"

dataset = ImageDataset(hr_folder, lr_folder)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# Load the RealESRGAN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(torch.load('weights/RealESRGAN_x4.pth'))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

# Loss function
# Load pre-trained VGG model for perceptual loss
vgg = vgg19(pretrained=True).features[:36].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# Initialize the combined loss function
criterion = CombinedLoss(vgg, device).to(device)

# Training loop
num_epochs = 50
loss_values = []

for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    for i, batch in enumerate(data_loader):
        hr_images = batch["hr"].to(device)
        lr_images = batch["lr"].to(device)

        optimizer.zero_grad()

        sr_images = model(lr_images)
        loss = criterion(sr_images, hr_images)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    average_loss = epoch_loss / num_batches
    loss_values.append(average_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

torch.save(model.state_dict(), "fine-tune-weights/RealESRGAN_x4.pth")

# Save loss values to a text file
with open("loss_values.txt", "w") as f:
    for loss_value in loss_values:
        f.write(f"{loss_value:.4f}\n")