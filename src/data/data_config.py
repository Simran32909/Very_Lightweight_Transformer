from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class RandomSynthDatasetConfig:
  _target_: str = field(default='src.data.data_config.RandomSynthDatasetConfig')
  name: str = field(default="")
  fonts_path: str = field(default="")
  max_len: int = field(default=20)
  words_to_generate: int = field(default=30000)

@dataclass
class SynthDatasetConfig:
  _target_: str = field(default='src.data.data_config.SynthDatasetConfig')
  name: str = field(default="")
  words_path: str = field(default="")
  fonts_path: str = field(default="")
  distr: bool = field(default=False)

@dataclass
class DatasetConfig:
  _target_: str = field(default='src.data.data_config.DatasetConfig')
  name: str = field(default="")
  images_path: str = field(default="")
  labels_path: str = field(default="")
  splits_path: str = field(default="")
  read_data: str = field(default='read_data')

@dataclass
class SharadaDatasetConfig:
  _target_: str = field(default='src.data.data_config.SharadaDatasetConfig')
  name: str = field(default="")
  data_dir: str = field(default="")
  split_file_path: str = field(default="")

@dataclass
class DataConfig:
  stage: str = field(default='train')
  img_size: tuple = field(default=(64, 1024))
  binarize: bool = field(default=True)
  batch_size: int = field(default=64)
  num_workers: int = field(default=4)
  pin_memory: bool = field(default=True)
  datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
  vocab_path: str = field(default='vocab.txt')
  transforms: List = field(default_factory=list)