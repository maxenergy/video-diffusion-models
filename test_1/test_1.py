# %% [markdown]
# # Video diffusion models

# %%
# Here wget old checkpoint
# !wget ...

# %%
# Params
image_size = 128
frames = 12
download_batch_size = 128
download_workers = 20

dataset_size = 125782

# If there is a checkpoint these changes automatically at runtime
epoch = 1
train_unet = 1
current_step = 0

# %%
import time
import os
start_time = time.time()

# Clean tmp files
for file in os.listdir("./"):
    if file.endswith('.gif'):
        os.remove(file)

# %% [markdown]
# ## Install dependencies

# %%
# !pip install imagen_pytorch==1.16.5

# %% [markdown]
# ## Utility functions to resize and crop GIFs

# %%
# GIF pre-processing

import numpy as np
from torchvision import transforms as T
from math import floor, fabs
from PIL import Image, ImageSequence


CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def center_crop(img, new_width, new_height): 
    width = img.size[0]
    height = img.size[1]
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))
    return img.crop((left, top, right, bottom))

def resize_crop_img(img, width, height):
    # width < height
    if( img.size[0] < img.size[1]):
        wpercent = (width/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((width, hsize), Image.Resampling.LANCZOS)
    else: # width >= height
        hpercent = (height/float(img.size[1]))
        wsize = int((float(img.size[0])*float(hpercent)))
        img = img.resize((wsize, height), Image.Resampling.LANCZOS)
    img = center_crop(img, width, height)
    # print(img.size[0])
    # print(img.size[1])
    return img

def transform_gif(img, new_width, new_height, frames, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]
    gif_frames = img.n_frames
    for i in range(0, frames):
        img.seek(i % gif_frames)
        img_out = resize_crop_img(img, new_width, new_height)
        yield img_out.convert(mode)
        
# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, fps = 10, loop = 0, optimize = True):
    print("Converting video tensors to GIF")
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    print(1000/fps)
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = int(1000/fps), loop = loop, optimize = optimize)
    print("Gif saved")
    return images

# gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path, width = 256, height = 256, frames = 32, channels = 3, transform = T.ToTensor()):
    print("Converting GIF to video tensors")
    img = Image.open(path)
    imgs = transform_gif(img, new_width = width, new_height = height, frames = frames, channels = channels)
    tensors = tuple(map(transform, imgs))
    return torch.stack(tensors, dim = 1)

# %% [markdown]
# ## Utility functions to download dataset

# %%
import os
import torch
import shutil
import urllib

from concurrent.futures import ThreadPoolExecutor, wait
import time
import threading

train_url = "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv"
train_data = "./train_data.tvs"

current_step = 0
texts = []
list_videos = []


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    try:
        #print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                    ' Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)

def get_videos(index_start, index_end):
    global texts
    global list_videos
    
    texts = []
    list_videos = []

    with open("train_data.tvs") as fp:
        for i, line in enumerate(fp):
            if i >= index_start and i< index_end :
                try:
                    file_img, file_text = line.split("\t")
                    print(f"Downloading image {i}")
                    download_url(file_img, "./", "download.gif")
                    tensor = gif_to_tensor('download.gif', width = image_size, height = image_size, frames = frames)
                    list_videos.append(tensor)
                    file_text = file_text[:-1] # Remove \n
                    texts.append(file_text)
                    os.remove('download.gif')
                except Exception as ex:
                    print(ex)
                    pass
            elif i > index_end:
                break

lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=download_workers)

def download_process_parallel(index, file_img, file_text):
    try:
        print(f"Downloading image {index}")
        download_url(file_img, "./", f"{index}.gif")
        tensor = gif_to_tensor(f"{index}.gif", width = image_size, height = image_size, frames = frames)
        file_text = file_text[:-1] # Remove \n
        with lock:
            list_videos.append(tensor)
            texts.append(file_text)
        os.remove(f"{index}.gif")
    except Exception as ex:
        print(ex)
        pass
    
def get_videos_parallel(index_start, index_end):
    global texts
    global list_videos
    
    texts = []
    list_videos = []

    with open("train_data.tvs") as fp:
        futures = []
        for i, line in enumerate(fp):
            if i >= index_start and i< index_end :
                file_img, file_text = line.split("\t")
                future = executor.submit(download_process_parallel, i, file_img, file_text)
                futures.append(future)
            elif i > index_end:
                break
        wait(futures)
                
def get_next_videos():
    global current_step
    get_videos_parallel(current_step, current_step + download_batch_size)
    current_step += len(texts)

# Download train data file
if not os.path.exists(train_data):
    download_url(train_url, "./", train_data)

# %% [markdown]
# ## Utility functions to save and load checkpoints

# %%
import shutil
import torch
import time
import gc
import os
from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from imagen_pytorch.data import Dataset

checkpoints_path = "./"
checkpoint_path = ""

def save_checkpoint(trainer: ImagenTrainer):
    global checkpoint_path
    print("Saving checkpoint")
    current_time = int(time.time())
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    checkpoint_path = os.path.join(checkpoints_path, f"checkpoint-unet_{train_unet}-epoch_{epoch}-step_{current_step}-{current_time}.pt")
    trainer.save(checkpoint_path)

def update_config(checkpoint):
    global epoch
    global train_unet
    global current_step
    splitted = (checkpoint.replace(".pt", "").split("checkpoint-")[1]).split("-")
    print(splitted)
    train_unet = int(splitted[0].replace("unet_", ""))
    epoch = int(splitted[1].replace("epoch_", ""))
    current_step = int(splitted[2].replace("step_", ""))
    print("Loaded configuration")
    print(f"Step: {current_step}")
    print(f"Epoch: {epoch}")
    print(f"Unet: {train_unet}")
    if current_step >= dataset_size:
        if train_unet == 2:
            print("End of training Unet 2 -> New epoch")
            train_unet = 1
            epoch += 1
        else:
            print("End of training Unet 1")
            train_unet = 2
        current_step = 0
        print("New configuration")
        print(f"Step: {current_step}")
        print(f"Epoch: {epoch}")
        print(f"Unet: {train_unet}")
        
def load_checkpoint(trainer: ImagenTrainer):
    global checkpoint_path
    timestamp = -1
    for file in os.listdir(checkpoints_path):
        if file.endswith('.pt'):
            new_timestamp = int((file.split("-")[4]).replace(".pt", ""))
            if new_timestamp > timestamp:
                checkpoint_path = os.path.join(checkpoints_path, file)
                timestamp = new_timestamp        
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found -> starting from scratch")
        return None
    trainer.load(checkpoint_path)
    update_config(checkpoint_path)

# %%
unet1 = Unet3D(
    dim = 64,
    cond_dim = 128,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet3D(
    dim = 64,
    cond_dim = 128,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

imagen = ElucidatedImagen(
    unets = (unet1, unet2),
    image_sizes = (16, 64),
    random_crop_sizes = (None, 16),
    num_sample_steps = (64, 64),
    # timesteps = 1000,
    cond_drop_prob = 0.1,                       # gives the probability of dropout for classifier-free guidance.
    sigma_min = 0.002,                          # min noise level
    sigma_max = (80, 160),                      # max noise level, double the max noise level for upsampler
    sigma_data = 0.5,                           # standard deviation of data distribution
    rho = 7,                                    # controls the sampling schedule
    P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
    P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn = 80,                               # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
).cuda()

trainer = ImagenTrainer(imagen)


# %% [markdown]
# ## Train Unet

# %%
# Train 
load_checkpoint(trainer)
counter = 0
while True:
    # if execution time is more than 11h40m stops
    if time.time() - start_time > 42000:
        break
    get_next_videos()
    # if there are no more videos stops
    if len(texts) == 0:
        break
    print("Generating tensor from videos")
    videos = torch.stack(list_videos, dim = 0).cuda()
    print(f"Training Unet {train_unet}")
    trainer(videos, texts = texts, unet_number = train_unet, max_batch_size = 32)
    trainer.update(unet_number = train_unet)
    del videos
    
    if counter % 100 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        print("Allocated memory")
        print(torch.cuda.memory_allocated())
#         save_checkpoint(trainer)
    counter +=1
save_checkpoint(trainer)

# %%
# !pip install GPUtil

# from GPUtil import showUtilization as gpu_usage
# gpu_usage()    

# %%
#end


