import matplotlib.pyplot as plt
import glob
import imageio
import PIL
import os

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False
    # This is so all layers run in inference mode (batchnorm)
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

        plt.savefig('./images/image_at_epoch{:04d}.png'.format(epoch))

# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('./images/image_at_epoch_{:04d}.png'.format(epoch_no))


# Use imageio to create an animated gif using the images saved during training
def animated_git(anim_file='dgcan.fig')
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('./images/image*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
