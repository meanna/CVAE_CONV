import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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


############################
#    Image Interpolation   #
############################


def interpolation(target_images, imgs, labels, model):
    """Performs a vector interpolation in the latent space to generate new images."""

    z_vectors = []
    resized_labels = []
    images = []

    # Computing the mean latent vector associated to each image
    for i in target_images:
        img = imgs[i][np.newaxis, ...]
        label = labels[i][np.newaxis, ...]
        model_output = model((img, label), is_train=False)
        img_z = model_output['z_mean']
        # img_var = model_output['z_log_var']
        z_vectors.append(img_z)
        resized_labels.append(label)

    for i in range(4):
        ratios = np.linspace(0, 1, num=8)
        vectors = []

        # Vectors interpolation
        for ratio in ratios:
            v = (1.0 - ratio) * z_vectors[i] + ratio * z_vectors[i + 1]
            vectors.append(v)

        vectors = np.asarray(vectors)

        # Generation
        for j, v in enumerate(vectors):
            if j < 4:
                z_cond = tf.concat([v, resized_labels[i]], axis=1)
                logits = model.decoder(z_cond, is_train=False)
                generated = tf.nn.sigmoid(logits)

            else:
                z_cond = tf.concat([v, resized_labels[i + 1]], axis=1)
                logits = model.decoder(z_cond, is_train=False)
                generated = tf.nn.sigmoid(logits)

            images.append(generated.numpy()[0, :, :, :])

    return images
