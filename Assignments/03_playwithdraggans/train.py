import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from GAN_network import   Generator,Discriminator,GAN
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model, dataloader, optimizer_g, optimizer_d, criterion_g, criterion_d, device, epoch, num_epochs):
    model.train()
    running_loss_g = 0.0
    running_loss_d = 0.0

    for i, (real_images, semantic_images) in enumerate(dataloader):
        
        real_images = real_images.to(device)
        semantic_images = semantic_images.to(device)
        #print(real_images.shape)
        #print(semantic_images.shape)
        # 训练判别器
        optimizer_d.zero_grad()
        # 真实图像的判别器损失
        real_combined = torch.cat((semantic_images, real_images), 1)
        real_outputs = model.discriminator(real_combined)
        real_loss = criterion_d(real_outputs, torch.ones_like(real_outputs))

        # 生成图像的判别器损失
        generated_images = model.generator(semantic_images)
        fake_combined = torch.cat((semantic_images, generated_images), 1)
        fake_outputs = model.discriminator(fake_combined.detach())
        fake_loss = criterion_d(fake_outputs, torch.zeros_like(fake_outputs))
        if epoch % 5 == 0 :
            save_images(semantic_images, real_images, generated_images, 'train_results', epoch)
        # 总判别器损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()

        # 生成器损失，包括对抗损失和 L1 损失
        fake_outputs =  model.discriminator(fake_combined)
        g_loss_gan = criterion_d(fake_outputs, torch.ones_like(fake_outputs))
        g_loss_l1 = criterion_g(generated_images, real_images)
        g_loss = g_loss_gan + 7 * g_loss_l1  # L1 损失通常乘以一个权重

        g_loss.backward()
        optimizer_g.step()

        running_loss_g += g_loss.item()
        running_loss_d += d_loss.item()
        
        print(f"Batch {i}/{len(dataloader)}, G Loss: {g_loss.item()}, D Loss: {d_loss.item()}")

    # Save sample images every 5 epochs
     


def validate(model, dataloader, criterion_G, device, epoch, num_epochs):
    model.eval()
    val_loss_G = 0.0

    with torch.no_grad():
        for i, (real_images, semantic_images) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            semantic_images = semantic_images.to(device)

            # Generate fake images
            fake_images = model.generator(semantic_images)

            # Compute the L1 loss
            loss_G_l1 = criterion_G(fake_images, real_images)
            val_loss_G += loss_G_l1.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(semantic_images, real_images, fake_images, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss_G = val_loss_G / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss_G: {avg_val_loss_G:.4f}')


def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    # 定义超参数
    in_channels = 3  # 输入图像的通道数
    out_channels = 3  # 输出图像的通道数
    # 创建生成器和判别器的实例
    model=GAN()
    netG=Generator()
    netD=Discriminator()
    # 创建GAN模型的实例
    criterion_G = nn.L1Loss()
    criterion_D = nn.BCELoss()
    optimizer_G = optim.Adam(model.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    
    

    # Add a learning rate scheduler for decay
    scheduler_G = StepLR(optimizer_G, step_size=200, gamma=0.2)
    scheduler_D = StepLR(optimizer_D, step_size=200, gamma=0.2)
    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer_G, optimizer_D, criterion_G, criterion_D, device, epoch, num_epochs)
        #validate(model, val_loader, criterion_G, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_G.step()
        scheduler_D.step()
        # Save model checkpoint every 20 epochs
        
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')
        
        

if __name__ == '__main__':
    main()
