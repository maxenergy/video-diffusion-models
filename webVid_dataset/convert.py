import cv2
import os
import urllib.request
import urllib.error
import glob
import math
import csv
import shutil
import queue
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


gif_frames = 12
width = 128
height = 128
max_lines = 1000000

dataset_url = "http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv"
dataset_file = "./data.csv"

out_dir = "./out"
out_tsv = "./out/dataset.tsv"

cloud_base_url = "."


class BlockingThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, *, queue_size=0, **kwargs):
        super().__init__(**kwargs)
        self._work_queue = queue.Queue(queue_size)

def download_url(url, root, filename=None):
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                    ' Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)

def convert_mp4_to_jpgs(filename_path, out_dir):
    video_capture = cv2.VideoCapture(filename_path)
    still_reading, image = video_capture.read()
    frame_count = 0
    while still_reading and frame_count < gif_frames:
        cv2.imwrite(f"{out_dir}/frame_{frame_count:03d}.jpg", image)
        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1

def center_crop(img, new_width, new_height): 
    width = img.size[0]
    height = img.size[1]
    left = int(math.ceil((width - new_width) / 2))
    right = width - int(math.floor((width - new_width) / 2))
    top = int(math.ceil((height - new_height) / 2))
    bottom = height - int(math.floor((height - new_height) / 2))
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
    return img

def make_gif(frames_folder, out_path, width, height):
    images = glob.glob(f"{frames_folder}/*.jpg")
    images.sort()
    frames = [Image.open(image) for image in images]
    frames = list(map(lambda frame: resize_crop_img(frame, width, height), frames))
    frame_one = frames[0]
    frame_one.save(out_path, format="GIF", optimize=True, append_images=frames, save_all=True, duration=50, loop=0, quality=10)

def worker(id, video_url, gif_path, width, height):
    try:
        download_url(video_url, f"./{id}", "download.mp4")
        convert_mp4_to_jpgs(f"./{id}/download.mp4", f"./{id}")
        make_gif(f"./{id}", gif_path, width, height)
        shutil.rmtree(f"./{id}")
    except:
        pass

def compute_videos(input_dataset_file, output_dataset_file, output_base_url, limit = 100):
    executor = BlockingThreadPoolExecutor(max_workers=20, queue_size=50)
    with open(input_dataset_file, 'r', encoding='utf-8') as file_in:
        with open(output_dataset_file, 'w', encoding='utf-8') as file_out:
            csv_reader = csv.reader(file_in, delimiter=',')
            line_count = 0
            for row in tqdm (csv_reader, desc="Processing", unit=' lines'):
                if line_count != 0:
                    text = row[1].replace("\n", "")
                    url = row[5]
                    # print(f'{url}\t{text}')
                    file_out.write(f'{output_base_url}/{line_count}.gif\t{text}\n')
                    executor.submit(worker, line_count, url, f"./out/{line_count}.gif", width, height)
                if line_count == limit:
                    break
                line_count += 1

if not os.path.exists(dataset_file):
    download_url(dataset_url, "./", dataset_file)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

compute_videos(dataset_file, out_tsv, cloud_base_url, max_lines)