import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from celeba import CelebADataset
# from ConvolutionalCondVAE import ConvCVAE, Decoder, Encoder
import time
import numpy as np
from matplotlib import pyplot as plt
from image_generation_utils import attr_manipulation, image_generation, attr_manipulation_interpolation
from utils import batch_generator, convert_batch_to_image_grid, read_data, train_step, \
    convert_batch_to_image_grid_interpolation
from image_reconstruction_utils import image_reconstruction
from datetime import datetime

start_time_total = time.perf_counter()
tf.random.set_seed(2)
# ----------------------------------------------------------------------
# Training configuration
run_train = False
print("run train....", run_train)

# trained models :  "2022-07-30_14.36.29"
# set to None if you want to train a new model
pretrained_model = "2022-08-07_01.14.16"#"2022-07-30_14.36.29"# "2022-08-08_12.19.07"
checkpoint_path = None  # "./checkpoints/2022-07-30_14.36.29/model-5"

# full dataset = "./embeddings.csv", "embeddings_128.csv", "embeddings_32.csv"
dataset_size = None
embedding_path = "embeddings_128_random.csv"
n_epochs = 1
save_model_every = 3
encoder_concat_input_and_condition = True

latent_dim = 128
learning_rate = 0.001
batch_size = 32
test_size = 32
save_test_set = True  # True  # True: the test set image IDs and other useful information will be stored in a
# pickle file
# to further uses (e.g. Image_Generation.ipynb)
# ----------------------------------------------------------------------
# choose model: ori, attention, deeper
model_type = "ori"

if pretrained_model in ["2022-08-07_14.37.18"]:
    model_type = "deeper"
    encoder_concat_input_and_condition = True
    latent_dim = 128

elif pretrained_model in ["2022-08-07_14.33.03"]:
    encoder_concat_input_and_condition = False
    model_type = "ori"
    latent_dim = 128

#  best setting
elif pretrained_model in ["2022-07-30_14.36.29", "2022-08-07_01.14.16"]:
    encoder_concat_input_and_condition = True
    model_type = "ori"
    latent_dim = 128

elif pretrained_model in ["2022-08-08_12.19.07"]:  # best model archi + 256 latent dim
    encoder_concat_input_and_condition = True
    model_type = "ori"
    latent_dim = 256
    batch_size = 32

if model_type == "ori":
    # ori model, attention
    from ConvolutionalCondVAE import ConvCVAE, Decoder, Encoder
elif model_type == "attention":
    # attention model
    from vae_attention import ConvCVAE, Decoder, Encoder

elif model_type == "deeper":
    # attention model
    from ConvolutionalCondVAE_2 import ConvCVAE, Decoder, Encoder

print("model type", model_type)
# ----------------------------------------------------------------------
celeba_path = './input/CelebA/img_align_celeba/img_align_celeba/'
dataset = CelebADataset(test_size=test_size, batch_size=batch_size, save_test_set=save_test_set,
                        embedding_path=embedding_path, celeba_path=celeba_path)
print("dataset.train_size", dataset.train_size)

train_size = dataset.train_size

date_time_obj = datetime.now()
timestamp_str = date_time_obj.strftime("%Y-%m-%d_%H.%M.%S")
print('Current Timestamp : ', timestamp_str)
# ----------------------------------------------------------------------

# Hyper-parameters
label_dim = 512  # do not change this
image_dim = [64, 64, 3]
beta = 1

# Model
encoder = Encoder(latent_dim, concat_input_and_condition=encoder_concat_input_and_condition)
decoder = Decoder()
model = ConvCVAE(
    encoder,
    decoder,
    label_dim=label_dim,
    latent_dim=latent_dim,
    beta=beta,
    image_dim=image_dim)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# ----------------------------------------------------------------------


# Checkpoint path

if pretrained_model:
    checkpoint_name = pretrained_model
else:
    checkpoint_name = timestamp_str

checkpoint_root = os.path.join(".", "checkpoints", checkpoint_name)
checkpoint_prefix = "model"
save_prefix = os.path.join(checkpoint_root, checkpoint_prefix)

result_folder_main = "results"
if not os.path.exists(result_folder_main):
    os.mkdir(result_folder_main)

result_folder = os.path.join(".", result_folder_main, f"result_{checkpoint_name}")
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

# ----------------------------------------------------------------------
# Define the checkpoint

checkpoint = tf.train.Checkpoint(module=model)

if checkpoint_path:
    checkpoint.restore(checkpoint_path).expect_partial()
    print("Load checkpoint ...", checkpoint_path)
else:
    # Restore the latest checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_root)
    print("latest checkpoint", latest)  # ./checkpoints/2022-08-06_10.27.42/model-6

    if latest is not None:
        checkpoint.restore(latest).expect_partial()
        # ./checkpoints/model_test/model-6
    elif not run_train:
        print("No checkpoint found !")
        sys.exit()

# ----------------------------------------------------------------------
# Read test_data.pickle
test_data = read_data("./test_data")
# Build a batch of test images
batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv',
                            celeba_path=celeba_path)
images, labels = next(batch_gen)
print("image batch", images.shape)
print("label batch", labels.shape)


def plot_recon_images(epoch, save_folder=None):
    # Image reconstruction
    print("\n Plot reconstructed images ...")
    if save_folder:
        result_folder_ = save_folder
    else:
        result_folder_ = result_folder
    image_reconstruction(model, images, labels, epoch, save_path=result_folder_)


# ----------------------------------------------------------------------
def generate_image_given_text(target_attr=None, save_folder=None):
    """E.g. target_attr = a man with bushy eyebrows"""
    print("\n Plot images given a text prompt ...")
    if save_folder:
        result_folder_ = save_folder
    else:
        result_folder_ = result_folder
    image_generation(model, labels, target_attr=target_attr, save_path=result_folder_)


# ----------------------------------------------------------------------
train_losses = []
train_recon_errors = []
train_latent_losses = []
loss = []
reconstruct_loss = []
latent_loss = []
step_index = 0


def train():
    n_batches = int(dataset.train_size / batch_size)
    print("Number of epochs: {},  number of batches: {}".format(n_epochs, n_batches))
    # save log
    with open(os.path.join(result_folder, "params.txt"), "w") as f:
        f.write(f"encoder_concat_input_and_condition = {encoder_concat_input_and_condition}\n")
        f.write(f"learning_rate = {learning_rate}\n")
        f.write(f"train_size = {train_size}\n")
        f.write(f"batch_size = {batch_size} \n")
        f.write(f"dataset.train_size = {dataset.train_size}\n")
        f.write(f"label_dim = {label_dim}\n")
        f.write(f"image_dim = {image_dim}\n")
        f.write(f"latent_dim = {latent_dim}\n")
        f.write(f"beta = {beta}\n")
        f.write(f"model checkpoint = {save_prefix}\n\n")
    # Epochs Loop

    for epoch in range(n_epochs):
        dataset.shuffle()  # Shuffling

        # Train Step Loop
        for step_index, inputs in enumerate(dataset):

            # print("input", type(input)) # <class 'builtin_function_or_method'>

            total_loss, recon_loss, lat_loss = train_step(inputs, model, optimizer)
            train_losses.append(total_loss)
            train_recon_errors.append(recon_loss)
            train_latent_losses.append(lat_loss)

            if step_index + 1 == n_batches:
                break

        loss.append(np.mean(train_losses, 0))
        reconstruct_loss.append(np.mean(train_recon_errors, 0))
        latent_loss.append(np.mean(train_latent_losses, 0))

        print("Epoch %i: loss %0.4f | reconstr loss %0.4f | latent loss %0.4f"
              % (epoch + 1, loss[epoch], reconstruct_loss[epoch], latent_loss[epoch]))

        # Save progress every n epochs
        if (epoch + 1) % save_model_every == 0:
            plot_recon_images(epoch)
            checkpoint.save(save_prefix)
            print(f"Model saved epoch {epoch + 1}: {save_prefix}")

        with open(os.path.join(result_folder, "params.txt"), "a") as f:
            print("Epoch %i: loss %0.4f | reconstr loss %0.4f | latent loss %0.4f"
                  % (epoch + 1, loss[epoch], reconstruct_loss[epoch], latent_loss[epoch]), file=f)

    # Save the final model if not saved yet
    if (n_epochs) % save_model_every != 0:
        checkpoint.save(save_prefix)
        print("Final model saved:", save_prefix)

    exec_time_total = time.perf_counter() - start_time_total
    print(f"time total = {exec_time_total} sec ({exec_time_total / 60.0} min )\n")
    with open(os.path.join(result_folder, "params.txt"), "a") as f:
        f.write(f"\ntime total = {exec_time_total} sec ({exec_time_total / 60.0} min )\n")


# ----------------------------------------------------------------------


def plot_image_with_attr(target_attr=None, image_embed_factor=0.5, save_folder=None):
    print("\n plot_image_with_attr ...")

    # Get reconstructed and modified images
    reconstructed_images, modified_images = attr_manipulation(images, labels, target_attr, model,
                                                              image_embed_factor=image_embed_factor)

    f = plt.figure(figsize=(20, 10))  # figsize=(64, 32)
    ax = f.add_subplot(1, 2, 1)
    ax.imshow(convert_batch_to_image_grid(reconstructed_images),
              interpolation='nearest')
    plt.axis('off')

    ax = f.add_subplot(1, 2, 2)
    ax.imshow(convert_batch_to_image_grid(modified_images),
              interpolation='nearest')
    plt.axis('off')
    str_target_attr = str(target_attr).replace(' ', '_')
    str_target_attr_factors = f"{str_target_attr}_{image_embed_factor}"
    file_name = f"modified_images_{str_target_attr_factors}.png"
    if save_folder:
        save_path = os.path.join(save_folder, file_name)
    else:
        save_path = os.path.join(result_folder, file_name)
    plt.title(str_target_attr_factors, fontsize=20, pad=10)
    plt.show()
    # plt.tight_layout()
    print(f"image is saved as {save_path}")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_attr_manipulation_interpolation(target_attr="wear glasses", num_images=1, save_folder=None):
    batch_result_list = attr_manipulation_interpolation(images, labels, target_attr, model, num_images)

    for i, result in enumerate(batch_result_list):

        f = plt.figure(figsize=(32, 40))  # figsize=(64, 32)

        ax = f.add_subplot(1, 2, 1)
        ax.imshow(convert_batch_to_image_grid_interpolation(result),
                  interpolation='nearest')
        plt.axis('off')

        str_target_attr = str(target_attr).replace(' ', '_')
        str_target_attr_factors = f"{str_target_attr}"
        file_name = f"modified_images_{str_target_attr_factors}_{i}.png"
        if save_folder:
            save_path = os.path.join(save_folder, file_name)
        else:
            save_path = os.path.join(result_folder, file_name)
        plt.title(str_target_attr_factors, fontsize=20, pad=10)
        plt.show()
        # plt.tight_layout()
        print(f"image is saved as {save_path}")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')  # dpi=100
        plt.close()


# ----------------------------------------------------------------------
def plot_ori_images(save_folder=None, i=0):
    # Plot original Images
    print("\n Plot original images")

    f = plt.figure(figsize=(32, 40))
    ax = f.add_subplot(1, 2, 1)
    ax.imshow(convert_batch_to_image_grid(images),
              interpolation='nearest')
    plt.axis('off')
    plt.title('original images', fontsize=20, pad=10)

    file_name = f"ori_images{i}.png"
    if save_folder:
        save_path = os.path.join(save_folder, file_name)
    else:
        save_path = os.path.join(result_folder, file_name)

    print(f"image is saved as {save_path}")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    if run_train:
        train()
    save_at = "./results/temp/"
    # save_at = result_folder <-- save to the model result folder

    # image reconstruction
    plot_recon_images(epoch=00, save_folder=save_at)
    #
    # # save original image
    plot_ori_images(save_folder=save_at, i=1)
    #
    # # 1) text-to-image
    generate_image_given_text(target_attr="wearing glasses and turn left", save_folder=save_at)
    generate_image_given_text(target_attr="smile", save_folder=save_at)
    # # plot in spectrum


    # 2) image-to-image
    generate_image_given_text(target_attr="smile", save_folder=save_at)

    # 3) attribute manipulation
    plot_image_with_attr(target_attr="wearing glasses", image_embed_factor=0.5, save_folder=save_at)



