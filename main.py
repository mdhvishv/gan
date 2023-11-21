import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import os
# Set random seed for reproducibility
manual_seed = 999
torch.manual_seed(manual_seed)
output_folder = "generated_images"
os.makedirs(output_folder, exist_ok=True)
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)

# Hyperparameters
num_epochs = 100
batch_size = 128
image_size = 64
latent_dim = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5

if __name__ == '__main__':
    # Define the data transformation and create the dataloader
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(root='cleb', transform=transform)# fill your directory of image with class
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # Define the generator
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            return self.main(input)

    # Define the discriminator
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    # Initialize the networks and optimizers
    netG = Generator()
    netD = Discriminator()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    netD.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    criterion = nn.BCELoss()

    # Create a batch of random noise for generating images
    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update the discriminator
            netD.zero_grad()
            real_data = data[0].to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), 1.0, device=device)  # Change label data type to float

            output = netD(real_data).view(-1)
            errD_real = criterion(output, label.float())  # Convert label to float
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_data = netG(noise)
            label.fill_(0.0)  # Change label data type to float
            output = netD(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label.float())  # Convert label to float
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()


            # Update the generator
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake_data).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Print training stats
            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        # Save generated images at fixed intervals
        if epoch % 10 == 0:
            fake = netG(fixed_noise)
            output_path = os.path.join(output_folder, f"fake_samples_epoch_{epoch:03d}.png")
            vutils.save_image(fake.detach(), output_path, normalize=True)

    # Save the trained models


    torch.save(netG.state_dict(), os.path.join(model_folder, "generator.pth"))
    torch.save(netD.state_dict(), os.path.join(model_folder, "discriminator.pth"))