import os
import random
import torch
import glob
import argparse
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from loss_function import CombinedLoss
from torchvision.models import vgg19
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter, Compose

parser = argparse.ArgumentParser(description="RealESRGAN Fine-Tuning")
parser.add_argument("--hr_folder", type=str, default="Dataset/HighResolution", help="Path to the high-resolution images folder")
parser.add_argument("--lr_folder", type=str, default="Dataset/LowResolution", help="Path to the low-resolution images folder")
parser.add_argument("--hr_crop_size", type=int, default=512, help="Random crop size for high-resolution images")
parser.add_argument("--lr_crop_size", type=int, default=128, help="Random crop size for low-resolution images")

args = parser.parse_args()

class ImageDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, hr_transforms, lr_transforms):
        self.hr_folder = hr_folder
        self.lr_folders = {
            "bicubic": os.path.join(lr_folder, "bicubic"),
            "nearest_neighbor": os.path.join(lr_folder, "nearest_neighbor"),
            "lanczos": os.path.join(lr_folder, "lanczos")
        }
        self.hr_images = sorted(glob.glob(f"{hr_folder}/*.*"))
        self.lr_images = {method: sorted(glob.glob(f"{folder}/*.*")) for method, folder in self.lr_folders.items()}
        
        self.hr_transforms = hr_transforms
        self.lr_transforms = lr_transforms

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_images[idx]).convert("RGB")
        lr_images = {}
        for method, img_paths in self.lr_images.items():
            lr_image = Image.open(img_paths[idx]).convert("RGB")
            lr_images[method] = lr_image

        if self.is_train:
            hr_image = self.train_hr_transforms(hr_image)
            lr_images = {method: self.train_lr_transforms(img) for method, img in lr_images.items()}
        else:
            hr_image = self.val_transforms(hr_image)
            lr_images = {method: self.val_transforms(img) for method, img in lr_images.items()}

        return {"hr": hr_image, "lr": lr_images}

    def __len__(self):
        return len(self.hr_images)
    
# Define the split_dataset function
def split_dataset(hr_folder, lr_folder, train_ratio=0.8):
    dataset = ImageDataset(hr_folder, lr_folder, val_transforms, val_transforms)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = ImageDataset(hr_folder, lr_folder, train_hr_transforms, train_lr_transforms)
    train_dataset.indices = train_indices

    val_dataset = ImageDataset(hr_folder, lr_folder, val_transforms, val_transforms)
    val_dataset.indices = val_indices

    return train_dataset, val_dataset

# hr_folder = "Dataset/HighResolution"
# lr_folder = "Dataset/LowResolution"

train_hr_transforms = Compose([
    RandomCrop(args.hr_crop_size),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ToTensor()
])

train_lr_transforms = Compose([
    RandomCrop(args.lr_crop_size),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ToTensor()
])

val_transforms = ToTensor()

dataset = ImageDataset(args.hr_folder, args.lr_folder, args.hr_crop_size, args.lr_crop_size)
train_set, val_set = split_dataset(args.hr_folder, args.lr_folder, train_ratio=0.8)
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

# Early stopping setup
patience = 10
epochs_without_improvement = 0
best_val_loss = float("inf")

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

        for method, lr_images in batch["lr"].items():
            lr_images = lr_images.to(device)

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

            for method, lr_images in batch["lr"].items():
                lr_images = lr_images.to(device)

                sr_images = model(lr_images)
                loss = criterion(sr_images, hr_images)

                val_epoch_loss += loss.item()
                val_num_batches += 1

    val_average_loss = val_epoch_loss / val_num_batches
    val_loss_values.append(val_average_loss)

    # Early stopping check
    if val_average_loss < best_val_loss:
        best_val_loss = val_average_loss
        epochs_without_improvement = 0
        # Save the best model so far
        torch.save(model.state_dict(), "fine-tune-weights/RealESRGAN_x4_best.pth")
    else:
        epochs_without_improvement += 1

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_average_loss:.4f}, Val Loss: {val_average_loss:.4f}")

    # Stop training if patience is exceeded
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

torch.save(model.state_dict(), "fine-tune-weights/RealESRGAN_x4.pth")

# Save loss values to a text file
with open("train_loss_values.txt", "w") as f:
    for loss_value in train_loss_values:
        f.write(f"{loss_value:.4f}\n")

with open("val_loss_values.txt", "w") as f:
    for loss_value in val_loss_values:
        f.write(f"{loss_value:.4f}\n")