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
- run `bash install.sh`
- run `python dataloader.py`
- the data will be stored in `input` folder

## generate CLIP embeddings for the dataset

- go to `src/generate_clip_embedings.py`, go to the main function set the number of images that you 
want to create embeddings for. If you want to do it for the whole dataset set `size=None`
- You will get `embeddings.csv` (1.7 GB)
- A small version is already included in the repo
  - `src/embeddings_128.csv`

## download model checkpoint

- model name = "2022-07-30_14.36.29"
- gdown https://drive.google.com/drive/folders/1YqXtAGVd2smI_VY8Vil0MOt_xyoqsboC?usp=sharing
- put it in "checkpoints" folder


## train the model
- go to `src/train.py`
- set `n_epochs`
- set `pretrained_model = "2022-07-30_14.36.29"`, set to `None` if you want to train a new model
- set `embedding_path = "embeddings_128.csv"`, (or full dataset = "embeddings.csv")
- go to the main function below, choose the functions you want to run (you can run all)
  - `train()`
    - the model will be saved under "checkpoints"
  - `generate_image_given_text(target_attr="wear reading glasses and smile")`
  - `plot_recon_images()`


## etc
- check the gpu run `src/check_gpu.py`