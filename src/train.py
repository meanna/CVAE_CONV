import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from celeba import CelebADataset
from ConvolutionalCondVAE import ConvCVAE, Decoder, Encoder
import time
from utils import train_step
import numpy as np
from image_generation_utils import interpolation
from matplotlib import pyplot as plt
from image_generation_utils import attr_manipulation
from utils import batch_generator, convert_batch_to_image_grid, read_data
from image_generation_utils import image_generation, image_reconstruction
from datetime import datetime

# Training configuration
# trained models :  "2022-07-30_14.36.29"
pretrained_model = "2022-07-30_14.36.29"  # set to None if you want to train a new model
# full dataset = "./embeddings.csv"
embedding_path = "embeddings_128.csv"
n_epochs = 6

learning_rate = 0.001
train_size = 0.5  # 0.01
batch_size = 32
save_test_set = True  # True: the test set image IDs and other useful information will be stored in a pickle file
# to further uses (e.g. Image_Generation.ipynb)

dataset = CelebADataset(train_size=train_size, batch_size=batch_size, save_test_set=save_test_set,
                        embedding_path=embedding_path)
print("dataset.train_size", dataset.train_size)

date_time_obj = datetime.now()
timestamp_str = date_time_obj.strftime("%Y-%m-%d_%H.%M.%S")
print('Current Timestamp : ', timestamp_str)
# ----------------------------------------------------------------------

# Hyper-parameters
label_dim = 512  # 40
image_dim = [64, 64, 3]
latent_dim = 128
beta = 0.65

# Model
encoder = Encoder(latent_dim)
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
    checkpoint_name = f"{timestamp_str}"
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
if not False:

    # Restore the latest checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_root)

    if latest is not None:
        checkpoint.restore(latest)
        print("Checkpoint restored:", latest)
    else:
        print("No checkpoint!")


# ----------------------------------------------------------------------
def plot_recon_images():
    # Read test_data.pickle
    test_data = read_data("./test_data")

    # Image reconstruction
    image_reconstruction(model, test_data, save_path=result_folder)


# ----------------------------------------------------------------------
def generate_image_given_text(target_attr=None):
    """E.g. target_attr = a man with bushy eyebrows"""
    # Read test_data.pickle
    test_data = read_data("./test_data")
    image_generation(model, test_data, target_attr=target_attr, save_path=result_folder)


# ----------------------------------------------------------------------
train_losses = []
train_recon_errors = []
train_latent_losses = []
loss = []
reconstruct_loss = []
latent_loss = []

step_index = 0
n_batches = int(dataset.train_size / batch_size)

print("Number of epochs: {},  number of batches: {}".format(n_epochs, n_batches))


# ----------------------------------------------------------------------
def train():
    # Epochs Loop
    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        dataset.shuffle()  # Shuffling

        # Train Step Loop
        for step_index, inputs in enumerate(dataset):

            # print("input", type(input)) # <class 'builtin_function_or_method'>

            total_loss, recon_loss, lat_loss = train_step(inputs, model, optimizer)
            train_losses.append(total_loss)
            train_recon_errors.append(recon_loss)
            train_latent_losses.append(lat_loss)

            if step_index + 1 == n_batches:
                # pass
                break

        loss.append(np.mean(train_losses, 0))
        reconstruct_loss.append(np.mean(train_recon_errors, 0))
        latent_loss.append(np.mean(train_latent_losses, 0))

        exec_time = time.perf_counter() - start_time
        print("Execution time: %0.3f \t Epoch %i: loss %0.4f | reconstr loss %0.4f | latent loss %0.4f"
              % (exec_time, epoch, loss[epoch], reconstruct_loss[epoch], latent_loss[epoch]))

        # Save progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            # plot_recon_images()
            print("epoch", epoch)
            # checkpoint.save(save_prefix + "_" + str(epoch + 1))
            checkpoint.save(save_prefix)
            print("Model saved:", save_prefix)
            with open(os.path.join(result_folder, "params.txt"), "w") as f:
                f.write(f"n_epochs = {epoch}\n")
                f.write(f"learning_rate = {learning_rate}\n")
                f.write(f"train_size = {train_size}\n")
                f.write(f"batch_size = {batch_size} \n")
                f.write(f"dataset.train_size = {dataset.train_size}\n")
                f.write(f"label_dim = {label_dim}\n")
                f.write(f"image_dim = {image_dim}\n")
                f.write(f"latent_dim = {latent_dim}\n")
                f.write(f"beta = {beta}\n")
                # add processing time
                f.write(f"model checkpoint = {save_prefix}\n")
                print("Execution time: %0.3f \t Epoch %i: loss %0.4f | reconstr loss %0.4f | latent loss %0.4f"
                      % (exec_time, epoch, loss[epoch], reconstruct_loss[epoch], latent_loss[epoch]), file=f)

    # Save the final model
    checkpoint.save(save_prefix)
    print("Model saved:", save_prefix)


# ----------------------------------------------------------------------


def plot_losses():
    f = plt.figure()
    plt.plot(reconstruct_loss, 'g', marker='o')
    plt.grid()
    plt.show()
    save_path = os.path.join(result_folder, "reconstruct_loss.png")
    plt.savefig(save_path, dpi=f.dpi)

    f = plt.figure()
    plt.plot(latent_loss, 'b', marker='o')
    plt.grid()
    plt.show()
    save_path = os.path.join(result_folder, "latent_loss.png")
    plt.savefig(save_path, dpi=f.dpi)

    f = plt.figure()
    plt.plot(loss, 'r', marker='o')
    plt.grid()
    plt.show()
    save_path = os.path.join(result_folder, "loss.png")
    plt.savefig(save_path, dpi=f.dpi)


# ----------------------------------------------------------------------

def plot_image_with_attr():
    # Read test_data.pickle
    test_data = read_data("./test_data")
    # Build a batch of test images
    batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv')
    images, labels = next(batch_gen)
    print("labels from batch generator", labels)

    # Dictionary with desired attributes with their value
    target_attr = {15: 1}

    # Get reconstructed and modified images
    reconstructed_images, modified_images = attr_manipulation(images, labels, target_attr, model)

    # ----------------------------------------------------------------------

    f = plt.figure(figsize=(64, 40))
    ax = f.add_subplot(1, 2, 1)
    ax.imshow(convert_batch_to_image_grid(reconstructed_images),
              interpolation='nearest')
    plt.axis('off')
    save_path = os.path.join(result_folder, "reconstructed_images.png")
    plt.savefig(save_path, dpi=200)

    ax = f.add_subplot(1, 2, 2)
    ax.imshow(convert_batch_to_image_grid(modified_images),
              interpolation='nearest')
    plt.axis('off')
    plt.show()
    save_path = os.path.join(result_folder, "modified_images.png")
    plt.savefig(save_path, dpi=200)


# ----------------------------------------------------------------------
def plot_ori_images():
    test_data = read_data("./test_data")
    # Build a batch of test images
    batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv')
    imgs, labels = next(batch_gen)

    # Plot original Images
    f = plt.figure(figsize=(32, 40))
    ax = f.add_subplot(1, 2, 1)
    ax.imshow(convert_batch_to_image_grid(imgs),
              interpolation='nearest')
    plt.axis('off')
    save_path = os.path.join(result_folder, "ori_images.png")
    plt.savefig(save_path, dpi=200)


# ----------------------------------------------------------------------
def plot_interpolation():
    test_data = read_data("./test_data")
    # Build a batch of test images
    batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv')
    imgs, labels = next(batch_gen)
    # Target images to interpolate
    target_images = [2, 12, 14, 22, 28]

    # Interpolation
    images = interpolation(target_images, imgs, labels, model)

    # Plot resulting images
    f = plt.figure(figsize=(32, 40))
    ax = f.add_subplot(1, 2, 1)
    ax.imshow(convert_batch_to_image_grid(np.asarray(images)))
    plt.axis('off')
    save_path = os.path.join(result_folder, "interpolation.png")
    plt.savefig(save_path, dpi=200)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    # train()
    # generate_image_given_text(target_attr="wear reading glasses and smile")
    # plot_recon_images()

    print('model name = ', checkpoint_name)
    print('result folder = ', result_folder)
