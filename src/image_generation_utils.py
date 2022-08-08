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
import random


def sample(z_mean, z_log_var, input_label, latent_dim=128):
    """ Performs the re-parametrization trick"""
    batch = input_label.shape[0]
    print("batch", batch)  # 32

    # eps = tf.random.normal(shape=(batch, latent_dim), mean=0.0, stddev=1.0)
    # z = z_mean + tf.math.exp(z_log_var * .5) * eps
    z_list = []
    for i in range(batch):
        mu = random.uniform(0.0, 1.0)
        log_var = random.uniform(0.0, 1.0)
        eps = tf.random.normal(shape=([latent_dim]), mean=0.0, stddev=1.0)
        z = mu + tf.math.exp(log_var * .5) * eps
        z_list.append(z)
    z_batch = tf.stack(z_list)
    # z = z_mean + tf.math.exp(z_log_var * .5) * eps_batch
    print("z", z_batch.shape)  # (32, 128)
    z_cond = tf.concat([z_batch, input_label], axis=1)  # (batch_size, label_dim + latent_dim)

    return z_cond


def image_generation(model, labels, target_attr=None, save_path=None):
    """
    Generates and plots a batch of images with specific attributes (if given).

    - list target_attr : list of desired attributes [default None]
    """

    # Vector of user-defined attributes.
    if target_attr:

        text = clip.tokenize([target_attr]).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text)

        text_condition = np.tile(text_features.cpu(), reps=[labels.shape[0], 1])
        print("Generation of 16 images with attributes: ", target_attr)

        # batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv',
        #                             celeba_path=celeba_path)
        # _, labels_image = next(batch_gen)
        # do interpolation
        #condition = (text_condition *0.5 ) + (labels*0.5) # labels is image embedding condition
        condition = text_condition
    else:
        condition = labels
        target_attr = "no text prompt"

    # z_cond = model.reparametrization(input_label=labels, z_mean=1.0, z_log_var=0.3)
    z_cond = sample(z_mean=1.0, z_log_var=0.3, input_label=condition, latent_dim=128)
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
        plt.savefig(os.path.join(save_path, "generation_" + prompt + ".png"), dpi=200, bbox_inches='tight')

    plt.show()
    plt.clf()


def image_generation_old(model, test_data, target_attr=None, save_path=None,
                     celeba_path='./input/CelebA/img_align_celeba/img_align_celeba/'):
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

        # batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv',
        #                             celeba_path=celeba_path)
        # _, labels_image = next(batch_gen)
        # labels = (labels *0.5 ) + (labels_image*0.5)

    # Vector of attributes taken from the test set.
    else:
        batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv',
                                    celeba_path=celeba_path)
        _, labels = next(batch_gen)
        print("Generation of 16 images with fixed attributes.", labels.shape)  # (32, 512)
        target_attr = "no attribute given"

    # z_cond = model.reparametrization(input_label=labels, z_mean=1.0, z_log_var=0.3)
    z_cond = sample(z_mean=1.0, z_log_var=0.3, input_label=labels, latent_dim=128)
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
        plt.savefig(os.path.join(save_path, "generation_" + prompt + ".png"), dpi=200, bbox_inches='tight')

    plt.show()
    plt.clf()
###############################
#   Attributes Manipulation   #
###############################


def attr_manipulation(images, labels, target_attr, model, image_embed_factor=0.5):
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
                    text_features.cpu().detach().numpy() * (1.0 - image_embed_factor))  # (1, 512)

            # condition with only input image embeddings
            # modified_label = text_features.cpu()

        # modified_label = (1, 512)
        z_cond = tf.concat([img_z, modified_label], axis=1)
        logits = model.decoder(z_cond, is_train=False)
        generated = tf.nn.sigmoid(logits)
        modified_images.append(generated.numpy()[0, :, :, :])

    return np.asarray(reconstructed_images, dtype='float32'), np.asarray(modified_images, dtype='float32')


def attr_manipulation_interpolation(images, labels, target_attr, model, num_images=1):
    """ Reconstructs a batch of images with modified attributes (target_attr)."""

    result_batch = []
    if num_images < (images.shape[0]):
        num = num_images  # range(images.shape[0])
    else:
        num = images.shape[0]

    for i in range(num):
        # reconstructed_images = []
        modified_images = []
        for beta in reversed(range(0, 10)):
            image_embed_factor = beta * 0.1
            img = images[i][np.newaxis, ...]
            label = labels[i][np.newaxis, ...]
            model_output = model((img, label), is_train=False)
            img_z = model_output['z_mean']

            # reconstructed_images.append(model_output['recon_img'].numpy()[0, :, :, :])
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
                        text_features.cpu().detach().numpy() * (1.0 - image_embed_factor))  # (1, 512)

                # condition with only input image embeddings
                # modified_label = text_features.cpu()

            # modified_label = (1, 512)
            z_cond = tf.concat([img_z, modified_label], axis=1)
            logits = model.decoder(z_cond, is_train=False)
            generated = tf.nn.sigmoid(logits)
            modified_images.append(generated.numpy()[0, :, :, :])
        result = np.asarray(modified_images, dtype='float32')  # (10, 64, 64, 3)
        result_batch.append(result)
        # break
    return result_batch

    # return np.asarray(reconstructed_images, dtype='float32'), np.asarray(modified_images, dtype='float32')
