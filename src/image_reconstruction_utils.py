import os

import matplotlib.pyplot as plt

from utils import convert_batch_to_image_grid


#########################
#  Image Reconstruction #
#########################


def image_reconstruction(model, images, labels, epoch=None, save_path=None):
    """
    Reconstructs and plots a batch of test images.
    """
    model_output = model((images, labels), is_train=False)

    f = plt.figure(figsize=(64, 40))
    ax = f.add_subplot(1, 2, 1)
    ax.imshow(convert_batch_to_image_grid(images))
    plt.title("original images", fontsize=60, pad=20)
    plt.axis('off')

    ax = f.add_subplot(1, 2, 2)
    ax.imshow(convert_batch_to_image_grid(model_output['recon_img'].numpy()))
    plt.axis('off')
    plt.title("reconstructed images", fontsize=60, pad=20)

    if save_path:
        file_name = "reconstruction.png"
        if epoch:
            file_name = f"reconstruction_{epoch}.png"
        plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')

    plt.show()
    plt.clf()

    print("Reconstruction of a batch of test set images.")
