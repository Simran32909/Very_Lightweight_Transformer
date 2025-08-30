# Datamodules for each Dataset
import torch
import os
import numpy as np
import torchvision
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List
from typing import Tuple
from fontTools.ttLib import TTFont
import cv2
import re
import random as rnd

from unidecode import unidecode

PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 2, 0, 1, 3

def read_htr_fonts(fonts_path):
  fonts = []
  for font in os.listdir(fonts_path):
    fonts.append(fonts_path + font)
  return fonts
    
def collate_fn(batch, img_size, text_transform):
    sequences_batch = []
    target_lengths = []  # To store the length of each sequence
    images_shapes = torch.tensor([image_sample.shape for image_sample, seq_sample in batch]) # Get shapes of images in batch [B, C, H, W]
    all_height_ratios = (images_shapes[:, 1] / img_size[0]) # Get height ratios for all images in batch
    all_width_reescaled = images_shapes[:, 2] / all_height_ratios # Get width reescaled for all images in batch
    assert all_height_ratios.shape[0] == all_width_reescaled.shape[0], 'All height ratios and all width reescaled must have the same length'
    max_width = img_size[1] # Get max width ratio
    
    images_batch = torch.ones(len(batch), 3, img_size[0], max_width) # Reescaled height and width and white background
    padded_columns = torch.zeros(len(batch)) # Padded columns for each image
  

    for i, (image_sample, seq_sample) in enumerate(batch):
      # Resize image to fixed height
      height, width = img_size[0], all_width_reescaled[i].int()

      if all_width_reescaled[i] > max_width:
        width = max_width

      image_resized_height = torchvision.transforms.Resize((height, width), antialias=True)(image_sample)
      images_batch[i, :, :, :image_resized_height.shape[2]] = image_resized_height 

      # Calculate padding
      padding_added = max_width - image_resized_height.shape[2]
      # padded_columns.append(padding_added)
      padded_columns[i] = padding_added

      # print(f'Seq sample: {seq_sample}')
      tokenized_sequence = text_transform(seq_sample)
      sequences_batch.append(tokenized_sequence)
      target_lengths.append(len(tokenized_sequence))

    sequences_batch = pad_sequence(sequences_batch, padding_value=PAD_IDX)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    assert images_batch.shape[0] == sequences_batch.shape[1] == padded_columns.shape[0] == target_lengths.shape[0], "Batch size of images, sequences, and lengths should be equal"
    
    return images_batch, sequences_batch, target_lengths, padded_columns
  
def has_glyph(font, glyph):
    # print(font['cmap'])
    font = TTFont(font)
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False

def generate_image(sequence, font, background_color=(255, 255, 255), text_color=(0, 0, 0)):
    txt = sequence
    font_name = font
    font = ImageFont.truetype(font, 50)
    img_size = (1500, 1000)

    # Check if a font can generate all the characters
    for char in txt:
      if has_glyph(font_name, str(char)) is False:
        raise Exception(f'Font {font} cannot generate char {char}')

    # Generate white image
    img = Image.new("RGB", (img_size[0], img_size[1]), background_color)
    draw = ImageDraw.Draw(img)
    draw.text((img_size[1]//10, img_size[0]//4), txt, font=font, fill=text_color)
    text_bbox = draw.textbbox((img_size[1]//10, img_size[0]//4), txt, font=font)
    img = img.crop(text_bbox)

    # Check if image shapes are zero
    assert img.size[0] != 0 and img.size[1] != 0, f'Image shape is zero. Image shape: {img.size}. Sequence: {sequence}'

    return img

class Dilation(object):
  def __init__(self, kernel_size=3, iterations=1):
    self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    self.iterations = iterations

  def __call__(self, image):
    # First invert the image
    image = cv2.bitwise_not(np.array(image))
    image = cv2.dilate(image, self.kernel, iterations=self.iterations)
    image =  cv2.bitwise_not(image)
    image = Image.fromarray(image)
    return image

# Erosion class for transform using opencv
class Erosion(object):
  def __init__(self, kernel_size=3, iterations=1):
    self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    self.iterations = iterations

  def __call__(self, image):
    # First invert the image
    image = cv2.bitwise_not(np.array(image))
    image = cv2.erode(image, self.kernel, iterations=self.iterations)
    image = cv2.bitwise_not(image)
    image = Image.fromarray(image)
    return image
    
    # return cv2.erode(np.array(image), self.kernel, iterations=self.iterations)

class Binarization(object):
  def __init__(self):
    pass

  def __call__(self, image):
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # Binarize image with opencv Otsu algorithm
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image

def read_data_1msharada(splits_path: str, label_key: str = 'original_text') -> Tuple[List[str], List[str]]:
  """
  Read 1MSharada dataset using a split file containing a JSON array of JSON file paths.

  :param splits_path: Path to split file (train.json/val.json/test.json), each containing a list of JSON annotation file paths
  :param label_key: Key inside per-sample JSON for the ground-truth text (default: 'original_text')
  :return: (image_paths, texts)
  """
  import json

  image_paths, texts = [], []

  with open(splits_path, 'r', encoding='utf-8') as f:
    json_files = json.load(f)

  for json_file in json_files:
    try:
      with open(json_file, 'r', encoding='utf-8') as jf:
        data = json.load(jf)

      image_path = data.get('image_path', '')
      text = data.get(label_key, '')

      if image_path.startswith('./'):
        image_path = image_path.replace('./data/1MSharada/', '/scratch/tathagata.ghosh/datasets/1MSharada/')

      if os.path.exists(image_path) and isinstance(text, str) and len(text.strip()) > 0:
        image_paths.append(image_path)
        texts.append(text)
      else:
        # Skip missing files or empty labels silently to keep loader robust
        pass
    except Exception:
      # Skip malformed entries
      pass

  return image_paths, texts

class Degradations(object):
  def __init__(self, ink_colors: List[str], paths_backgrounds: str):
    # Colors come in a list of #RRGGBB strings in hexadecimal
    # conver to tuple of (R, G, B) for each color
    self.colors = [tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) for color in ink_colors]
    self.paths_backgrounds = paths_backgrounds
    # self.files_backgrounds = !find $paths_backgrounds -type f -name "*.png"
    # Same function but in python with os without using bash
    extensions = ['.png', '.jpg', '.jpeg']
    self.files_backgrounds = [os.path.join(dp, f) for dp, dn, filenames in os.walk(paths_backgrounds) for f in filenames if os.path.splitext(f)[1] in extensions]
    # print(f'Files {self.files_backgrounds}')

  def __call__(self, image):
    # Binarize with opencv to obtain a mask of the pixels
    image_thres = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(image_thres, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ink_image = Image.new('RGB', image.size, color=self.colors[np.random.randint(0, len(self.colors))])
    ink_image = ink_image.resize(image.size)
    image = Image.composite(image, ink_image, Image.fromarray(mask))

    background = cv2.imread(self.files_backgrounds[np.random.randint(0, len(self.files_backgrounds))])
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB) # Convert to RGB
    # Apply median filter to remove text
    background = cv2.medianBlur(background, 51)

    # Convert to PIL
    img_pil_background = Image.fromarray(background)
    img_pil_background_resized = img_pil_background.resize(image.size)

    # Composite image with background without pixels of the text (using mask)
    # invert mask to composite the image with the background
    image = Image.composite(img_pil_background_resized, image, Image.fromarray(mask))
    
    return image