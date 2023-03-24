import matplotlib.pyplot as plt
import numpy as np
import torch


def show_tensor_batch(tensors):
    # Create a figure and set the subplot layout
    fig, axes = plt.subplots(8, 8, figsize=(100, 100))

    # Loop over the tensors and display each one
    for i, tensor in enumerate(tensors):
        # Convert the PyTorch tensor to a numpy array
        numpy_img = tensor.detach().cpu().numpy()

        # Normalize the pixel values between 0 and 1
        numpy_img = (numpy_img - np.min(numpy_img)) / (np.max(numpy_img) - np.min(numpy_img))

        # Display the image in the appropriate subplot
        row = i // 8
        col = i % 8
        axes[row, col].imshow(numpy_img.transpose((1, 2, 0)))
        axes[row, col].axis('off')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Display the figure
    plt.show()
    
    
def show_tensor_comparison(original, inpainted, save=False, img_name="infill_eval.png"):
    # Create a figure and set the subplot layout
    fig, axes = plt.subplots(8, 8, figsize=(100, 100))

    # Loop over the tensors and display each one
    for i, (origin, inpaint) in enumerate(zip(original, inpainted)):
        # Convert the PyTorch tensor to a numpy array
        i = i*2
        
        numpy_img_org = origin.detach().cpu().numpy()

        numpy_img_inp = inpaint.detach().cpu().numpy()

        # Normalize the pixel values between 0 and 1
        numpy_img_org = (numpy_img_org - np.min(numpy_img_org)) / (np.max(numpy_img_org) - np.min(numpy_img_org))
        numpy_img_inp = (numpy_img_inp - np.min(numpy_img_inp)) / (np.max(numpy_img_inp) - np.min(numpy_img_inp))

        # Display the image in the appropriate subplot
        row = i // 8
        col = i % 8
        axes[row, col].imshow(numpy_img_org.transpose((1, 2, 0)))
        axes[row, col].axis('off')
        
        axes[row, col+1].imshow(numpy_img_inp.transpose((1, 2, 0)))
        axes[row, col+1].axis('off')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Display the figure
    plt.show()
    if save:
        plt.savefig('./pictures/' + img_name)



def evaluate_inpainter(inpainter, dataloader, save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inpainter.to(device)
    inpainter.eval()
    mask_tensor = torch.ones([3, 128, 128], requires_grad=False)
    mask_tensor[:, 32:96, 32:96] = 0
    mask_tensor.to(device)
    
    batch = next(iter(dataloader))[0]
#     show_tensor_batch(batch)
    overlap = 4
    
    #Get Center 64 x 64 of the 128 x 128 images
    center_imgs = batch[:, :, 32:96, 32:96].to(device)
    
    #Transform the imgs using the center mask
    masked_imgs = batch[:]
    masked_imgs = masked_imgs.to(device)
    masked_imgs[:, 0, 32+overlap:96-overlap, 32+overlap:96-overlap] = 2*117.0/255.0 - 1.0
    masked_imgs[:, 1, 32+overlap:96-overlap, 32+overlap:96-overlap] = 2*104.0/255.0 - 1.0
    masked_imgs[:, 2, 32+overlap:96-overlap, 32+overlap:96-overlap] = 2*123.0/255.0 - 1.0

    
    generated_imgs = inpainter(masked_imgs)
    
    #Infill the masked_images
    inpainted_imgs = masked_imgs.clone().to(device)
    inpainted_imgs[:,:, 32:96, 32:96] = generated_imgs
    
    show_tensor_comparison(batch[:64//2, :, :,:], inpainted_imgs[:64//2, :, :,:], save=save)
    
def show_pictures(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mask_tensor = torch.ones([3, 128, 128], requires_grad=False)
    mask_tensor[:, 32:96, 32:96] = 0
    mask_tensor.to(device)
    
    batch = next(iter(dataloader))[0]
#     show_tensor_batch(batch)
    
    
    #Get Center 64 x 64 of the 128 x 128 images
    center_imgs = batch[:, :, 32:96, 32:96].to(device)
    
    #Transform the imgs using the center mask
    masked_imgs = batch[:] * mask_tensor
    masked_imgs = masked_imgs.to(device)
    masked_imgs[:,:, 32:96, 32:96] = 1
    
    show_tensor_comparison(batch[:64//2, :, :,:], masked_imgs[:64//2, :, :,:])
    

