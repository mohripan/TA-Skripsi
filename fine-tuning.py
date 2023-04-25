import os
import torch
import glob
from torch.utils.data import DataLoader, Dataset, random_split
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
    
# Define the split_dataset function
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return train_set, val_set

hr_folder = "Dataset/HighResolution"
lr_folder = "Dataset/LowResolution"

dataset = ImageDataset(hr_folder, lr_folder)
train_set, val_set = split_dataset(dataset, train_ratio=0.8)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

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
train_loss_values = []
val_loss_values = []

for epoch in range(num_epochs):
    # Train the model
    model.train()
    train_epoch_loss = 0
    train_num_batches = 0
    for i, batch in enumerate(train_loader):
        hr_images = batch["hr"].to(device)
        lr_images = batch["lr"].to(device)

        optimizer.zero_grad()

        sr_images = model(lr_images)
        loss = criterion(sr_images, hr_images)
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
        train_num_batches += 1

    train_average_loss = train_epoch_loss / train_num_batches
    train_loss_values.append(train_average_loss)

    # Evaluate the model on the validation set
    model.eval()
    val_epoch_loss = 0
    val_num_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            hr_images = batch["hr"].to(device)
            lr_images = batch["lr"].to(device)

            sr_images = model(lr_images)
            loss = criterion(sr_images, hr_images)

            val_epoch_loss += loss.item()
            val_num_batches += 1

    val_average_loss = val_epoch_loss / val_num_batches
    val_loss_values.append(val_average_loss)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_average_loss:.4f}, Val Loss: {val_average_loss:.4f}")

torch.save(model.state_dict(), "fine-tune-weights/RealESRGAN_x4.pth")

# Save loss values to a text file
with open("train_loss_values.txt", "w") as f:
    for loss_value in train_loss_values:
        f.write(f"{loss_value:.4f}\n")

with open("val_loss_values.txt", "w") as f:
    for loss_value in val_loss_values:
        f.write(f"{loss_value:.4f}\n")