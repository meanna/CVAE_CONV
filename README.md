# Conditional VAE + CLIP for text to image generation

based on https://github.com/EleMisi/ConditionalVAE


## if using colab, install these packages

- pip install ftfy regex tqdm
- pip install git+https://github.com/openai/CLIP.git


## if not using colab, setup a conda env

- conda create -n tf -c conda-forge tensorflow-gpu python=3.7 &&conda activate tf
- pip install kaggle
- pip install pandas
- pip install matplotlib
- python3 -m pip install opencv-python
- pip install ftfy regex tqdm
- pip install git+https://github.com/openai/CLIP.git


## download the dataset
- add `kaggle.json` ([how to get this file](https://www.kaggle.com/general/156610)) to src
- cd to src
- run `bash setup_kaggle.sh`
- run `python dataloader.py`
- the data will be stored in `input` folder
- if your data is in another folder set the path at celeba.get_images
- if you use resized images, adjust celeba.preprocess_image because it will crop and resize the image again. 

## generate CLIP embeddings for the dataset

- go to `src/generate_clip_embedings.py`, go to the main function set the number of images that you 
want to create embeddings for. If you want to do it for the whole dataset set `size=None`
- alternatively, run `python src/generate_clip_embeddings_lm.py 128` with 128 as the desired number of embeddings. you may choose another number. if no number is specified, it will embed all images in the dataset (>202,000).
- You will get `embeddings.csv` (1.7 GB)
- A small version is already included in the repo
  - `src/embeddings_128.csv`

## download model checkpoint

- model name = "2022-07-30_14.36.29"
- `gdown https://drive.google.com/uc?id=1P1z0Jl_wND6mqR73QSZ59bDxPO7IEX1P`
- `unzip -q 2022-07-30_14.36.29.zip -d ./`
- put it in "checkpoints" folder, `mv 2022-07-30_14.36.29 checkpoints/`


## train the model
- go to `src/train.py`
- set `n_epochs = [number]`
- set `save_model_every = [number]`
- set `pretrained_model = "2022-07-30_14.36.29"`, set to `None` if you want to train a new model
- set `embedding_path = "embeddings_128.csv"`, (or full dataset = "embeddings.csv")
- set `run_train = True`
  - the model will be saved in `checkpoints` folder
- go to the main function below, choose the functions you want to run (you can run all)
  - `generate_image_given_text(target_attr="wear reading glasses and smile")`
  - `plot_recon_images()`
  - `plot_image_with_attr(target_attr="angry", image_embed_factor=0.6, new_attr_factor=0.8)`
  - `plot_interpolation()`
  - `plot_ori_images()`


## etc
- check the gpu run `src/check_gpu.py`
