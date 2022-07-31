from ast import literal_eval

import clip
import pandas as pd
import torch
from PIL import Image


def create_clip_embeddings(size=None):
    file_path = "./input/CelebA/list_attr_celeba.csv"
    df = pd.read_csv(file_path)
    if size:
        df = df[:size]
    new_df = df['image_id']
    new_df = new_df.reset_index()  # make sure indexes pair with number of rows

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.cuda()  # .eval()
    # input_resolution = model.visual.input_resolution
    # context_length = model.context_length
    # vocab_size = model.vocab_size
    #
    # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # print("Input resolution:", input_resolution)
    # print("Context length:", context_length)
    # print("Vocab size:", vocab_size)

    embeddings = []

    # new_df = new_df[:300]

    def my_gen():
        images = []
        for i, row in new_df.iterrows():
            image_id = row['image_id']
            image_path = './input/CelebA/img_align_celeba/img_align_celeba/' + image_id
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image = torch.squeeze(image)
            images.append(image)
            if i % 100 == 0:
                print(i)
                yield images
                images = []
        yield images

    for images in my_gen():
        images = torch.stack(images)
        # print(images.shape) # torch.Size([100, 3, 224, 224])
        with torch.no_grad():
            image_features = model.encode_image(images.to(device))

        for i, _ in enumerate(image_features):
            image_f = image_features[i].cpu().detach().tolist()
            embeddings.append(image_f)

    print("embeddings", len(embeddings))
    new_df["embeddings"] = embeddings
    # print(type(embeddings))
    # new_df["embeddings"] = new_df["embeddings"].astype(float)

    if size:
        out_file = "embeddings_" + str(size) + ".csv"
    else:
        out_file = "embeddings"
    new_df.to_csv(out_file)
    print("out_file", out_file)

# this is to check the result
def load_embeddings():
    df = pd.read_csv("embeddings.csv", index_col=0,
                     converters={'embeddings': literal_eval})
    print(df.columns)
    img_ids = dict(zip(df['image_id'], df["embeddings"].values))
    print(type(img_ids["000001.jpg"]))
    print(img_ids["000001.jpg"][10])


if __name__ == "__main__":
    create_clip_embeddings(size=128)
    # load_embeddings()

