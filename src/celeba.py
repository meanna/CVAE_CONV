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
        self.train_img_ids, self.test_img_ids, self.attributes = self.load(train_size)
        self.batch_size = batch_size
        self.mode = mode
        self.train_size = len(self.train_img_ids)

        if save_test_set:
            self.save_test_set()

    def load(self, train_dim):
        """ 
        Loads all image IDs and the attributes and splits the dataset into training set and test set.
            
            Returns:
                    - train_img_ids [list]
                    - test_img_ids [list]
                    - attributes [list]
            
        """

        print("Loading images id and attributes...")
        # "./input/CelebA/list_attr_celeba.csv"
        # "./embeddings_300.csv"
        file_path = self.embedding_path
        df = pd.read_csv(file_path, index_col=0, converters={'embeddings': literal_eval})
        # df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)

        # df = df[:200]

        attributes = [x for x in df.columns]  # + ["test"] # list(range(40))
        # print("attributes", attributes)
        # attributes[
        #     '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
        #     'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
        #     'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
        #     'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
        #     'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        #     'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        # od = OrderedDict(df.to_dict('index'))
        # #print(od)
        # img_ids = OrderedDict()
        # for k,v in od.items():
        #   img_id = [np.float32(x) for x in v.values()]
        #   img_ids[k] = img_id
        # print("img_ids: {} \nAttributes: {} \n".format(len(img_ids), len(attributes)))
        id_to_embed = OrderedDict()
        id_to_embed_temp = dict(zip(df['image_id'], df["embeddings"].values))
        # print(type(od["000001.jpg"])) # list
        # print(od["000001.jpg"][10])
        for k, v in id_to_embed_temp.items():
            img_id = np.float32(v)
            id_to_embed[k] = img_id
        # print(type(img_ids["000001.jpg"])) #<class 'numpy.ndarray'>

        # Splitting
        print("Splitting dataset...\n")
        n_train = int(len(id_to_embed) * train_dim)
        list_img_ids = list(id_to_embed.items())
        train_img_ids = list_img_ids[:n_train]
        test_img_ids = list_img_ids[n_train:]

        print("Train set dimension: {} \nTest set dimension: {} \n".format(len(train_img_ids), len(test_img_ids)))

        return train_img_ids, test_img_ids, attributes

    def next_batch(self, idx):
        """
        Returns a batch of images with their ID as numpy arrays.
        The first returned value is the input images with shape (batch, 64, 64, 3).
        The second returned value is the condition (batch, label_dim).
        """
        # batch_img_ids is the attribute vector
        batch_img_ids = [x[1] for x in self.train_img_ids[idx * self.batch_size: (idx + 1) * self.batch_size]]
        images_id = [x[0] for x in self.train_img_ids[idx * self.batch_size: (idx + 1) * self.batch_size]]
        batch_imgs = self.get_images(images_id)

        return np.asarray(batch_imgs, dtype='float32'), np.asarray(batch_img_ids, dtype='float32')

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
                'attributes': self.attributes,
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
