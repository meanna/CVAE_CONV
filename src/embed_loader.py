import math
import random
from ast import literal_eval
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from utils import save_data


class CelebADataset(Sequence):

    def __init__(self, train_size, batch_size, mode='train', save_test_set=False,
                 embedding_path="./embeddings_300.csv"):
        self.embedding_path = embedding_path
        self.train_img_ids, self.test_img_ids  = self.load(train_size)
        self.batch_size = batch_size
        self.mode = mode
        self.train_size = len(self.train_img_ids)

        if save_test_set:
            self.save_test_set()

    def load(self, test_size):
        """
        Loads all image IDs and the attributes and splits the dataset into training set and test set.

            Returns:
                    - train_img_ids [list]
                    - test_img_ids [list]
                    - attributes [list]

        """

        file_path = "./input/CelebA/list_attr_celeba.csv"
        df = pd.read_csv(file_path)
        image_ids = df['image_id']
        #image_ids = image_ids.reset_index()  # make sure indexes pair with number of rows
        #image_ids = image_ids[:128] #.values.tolist()
        #print(image_ids)

        # Splitting
        print("Dataset size", len(image_ids))
        print("Splitting dataset...\n")
        n_train = len(image_ids) - test_size  # int(len(id_to_embed) * train_dim)
        print("n_train", n_train)
        train_img_ids = image_ids[:n_train]
        test_img_ids = image_ids[n_train:]

        print("Train set dimension: {} \nTest set dimension: {} \n".format(len(train_img_ids), len(test_img_ids)))

        return train_img_ids, test_img_ids

    def next_batch(self, idx):
        """
        Returns a batch of images with their ID as numpy arrays.
        The first returned value is the input images with shape (batch, 64, 64, 3).
        The second returned value is the condition (batch, label_dim).
        """
        # batch_img_ids is the attribute vector
        #batch_img_ids = [x[1] for x in self.train_img_ids[idx * self.batch_size: (idx + 1) * self.batch_size]]
        images_id = [x for x in self.train_img_ids[idx * self.batch_size: (idx + 1) * self.batch_size]]
        batch_imgs = self.get_images(images_id)
        batch_embeds = self.get_embeddings(images_id)

        return np.asarray(batch_imgs, dtype='float32'), np.asarray(batch_embeds, dtype='float32')

    def get_embeddings(self, embed_ids):
        """
        Returns the list of images corresponding to the given IDs.
        """
        embeds = []

        for image_name in embed_ids:
            embed_name = image_name[:-4]
            embed_path = './embeddings/' + f"{embed_name}.npy"
            with open(embed_path, 'rb') as f:
                a = np.load(f, allow_pickle=True)
                embeds.append(a)

        return embeds

    def preprocess_image(self, image_path, img_size=128, img_resize=64, x=25, y=45):
        """
        Crops, resizes and normalizes the target image.
        """

        img = cv2.imread(image_path)
        img = img[y:y + img_size, x:x + img_size]
        img = cv2.resize(img, (img_resize, img_resize))
        img = np.array(img, dtype='float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img /= 255.0  # Normalization to [0.,1.]

        return img

    def get_images(self, imgs_id):
        """
        Returns the list of images corresponding to the given IDs.
        """
        imgs = []

        for i in imgs_id:
            image_path = './input/CelebA/img_align_celeba/img_align_celeba/' + i
            imgs.append(self.preprocess_image(image_path))

        return imgs

    def save_test_set(self):
        """
        Saves a json file with useful information for teh test phase:
            - training size
            - test images IDs
            - attributes
            - batch size
        """

        try:
            test_data = {
                'train_size': self.train_size,
                'test_img_ids': self.test_img_ids,
                'batch_size': self.batch_size
            }

            file_path = "./test_data"
            save_data(file_path, test_data)
        except:
            raise
        print("Test img_ids successfully saved.")

    def shuffle(self):
        """
        Shuffles the training IDs.
        """
        self.train_img_ids = random.sample(self.train_img_ids, k=self.train_size)
        print("IDs shuffled.")

    def __len__(self):
        return int(math.ceil(self.train_size / float(self.batch_size)))

    def __getitem__(self, index):
        return self.next_batch(index)


from ast import literal_eval

import clip
import pandas as pd
import torch
from PIL import Image

import os

if not os.path.exists("embeddings"):
    print("create...")
    os.mkdir("embeddings")

def get_image_ids(size=None):
    file_path = "./input/CelebA/list_attr_celeba.csv"
    df = pd.read_csv(file_path)
    if size:
        df = df[:size]
    image_ids = df['image_id']
    image_ids = image_ids.reset_index()  # make sure indexes pair with number of rows
    #image_ids = image_ids[:128]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.cuda()

    def my_gen():
        embed_file_name = []
        images = []
        for i, row in image_ids.iterrows():
            image_id = row['image_id']
            image_path = './input/CelebA/img_align_celeba/img_align_celeba/' + image_id
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image = torch.squeeze(image)
            images.append(image)
            embed_file_name.append(image_id)
            if i % 1000 == 0:
                print(i)
                print(len(images))
                yield images, embed_file_name
                images = []
                embed_file_name = []
        yield images, embed_file_name


    for images, embed_file_name in my_gen():
        images = torch.stack(images)
        # print(images.shape) # torch.Size([100, 3, 224, 224])
        with torch.no_grad():
            image_features = model.encode_image(images.to(device))

        for i in range((len(image_features))):
            image_f = image_features[i].cpu().detach().numpy()
            #with open(out_name, 'wb') as f:
            np.save(os.path.join(".","embeddings", f"{embed_file_name[i][:-4]}"), image_f)
            #np.save(f, image_f, allow_pickle=True)

    #return embed_file_name.to_list()


get_image_ids()
# print(r)


def load_numpy():
    with open(os.path.join(".", "embeddings", "000001.npy"), 'rb') as f:
        print(type(f))
        #for _ in range(10):
        a = np.load(f, allow_pickle=True)
        print(type(a))
        print(a.shape)

#load_numpy()

def f1():
    embedding_path = "embeddings_128.csv"
    n_epochs = 20
    save_model_every = 10
    encoder_concat_input_and_condition = False
    learning_rate = 0.001
    train_size = 0.5  # 0.01
    batch_size = 32
    save_test_set = False
    dataset = CelebADataset(train_size=train_size, batch_size=batch_size, save_test_set=save_test_set,
                            embedding_path=embedding_path)

    #print(dataset.train_img_ids)
    #print(dataset.test_img_ids)
    # from utils import batch_generator, read_data
    # test_data = read_data("./test_data")
    # batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name='Conv')
    # images, labels = next(batch_gen)

    for step_index, inputs in enumerate(dataset):
        print(type(inputs))
        #print(inputs.shape)
        print(inputs[1].shape) # (32, 512)
        print(inputs[0].shape) # (32, 64, 64, 3)


#f1()