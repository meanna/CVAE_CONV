import os

import clip
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

from utils import batch_generator, convert_batch_to_image_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
#################################
#  Conditional Image Generation #
#################################

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.cuda()  # .eval()


def image_generation(model, test_data, target_attr=None, save_path=None):
    """
    Generates and plots a batch of images with specific attributes (if given).

    - list target_attr : list of desired attributes [default None]
    """

    # Vector of user-defined attributes.
    if target_attr:

        text = clip.tokenize([target_attr]).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text)

        labels = np.tile(text_features.cpu(), reps=[test_data['batch_size'], 1])
        print("Generation of 16 images with attributes: ", target_attr)

    # Vector of attributes taken from the test set.
    else:
        batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv')
        _, labels = next(batch_gen)
        print("Generation of 16 images with fixed attributes.")
        target_attr = "no attribute given"

    z_cond = model.reparametrization(input_label=labels, z_mean=1.0, z_log_var=0.3)
    logits = model.decoder(z_cond, is_train=False)
    generated = tf.nn.sigmoid(logits)

    # Plot
    f = plt.figure(figsize=(10, 10))
    # plt.title(str(target_attr))
    ax = f.add_subplot(1, 1, 1)
    ax.imshow(convert_batch_to_image_grid(generated.numpy()))
    plt.axis('off')
    prompt = target_attr.replace(' ', '_')
    plt.title(prompt, fontsize=20, pad=20)

    if save_path:
        plt.savefig(os.path.join(save_path, "generation_" + prompt + ".png"), dpi=300, bbox_inches='tight')

    plt.show()
    plt.clf()


###############################
#   Attributes Manipulation   #
###############################


def attr_manipulation(images, labels, target_attr, model, image_embed_factor=1.0, new_attr_factor=1.0):
    """ Reconstructs a batch of images with modified attributes (target_attr)."""

    reconstructed_images = []
    modified_images = []

    for i in range(images.shape[0]):
        img = images[i][np.newaxis, ...]
        label = labels[i][np.newaxis, ...]
        model_output = model((img, label), is_train=False)
        img_z = model_output['z_mean']

        reconstructed_images.append(model_output['recon_img'].numpy()[0, :, :, :])
        if target_attr is None:
            modified_label = np.expand_dims(labels[i], axis=0)
        else:
            text = clip.tokenize([target_attr]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text)
            # labels_ = np.expand_dims(labels[i], axis=0)  # (1, 512) -- not needed
            # type(labels[i] #<class 'numpy.ndarray'> of shape (512,)
            # condition with input image embeddings + given text embeddings
            modified_label = (labels[i] * image_embed_factor) + (
                        text_features.cpu().detach().numpy() * new_attr_factor)  # (1, 512)

            # condition with only input image embeddings
            # modified_label = text_features.cpu()

        # modified_label = (1, 512)
        z_cond = tf.concat([img_z, modified_label], axis=1)
        logits = model.decoder(z_cond, is_train=False)
        generated = tf.nn.sigmoid(logits)
        modified_images.append(generated.numpy()[0, :, :, :])

    return np.asarray(reconstructed_images, dtype='float32'), np.asarray(modified_images, dtype='float32')
