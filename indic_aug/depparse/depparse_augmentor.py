from torch.utils.data import Dataset

from ..globals import Augmentor

class DepparseAugmentor(Augmentor, Dataset):
    def __init__(self, src_input_path, tgt_input_path, transform=True):
        self.src_input_path = src_input_path
        self.tgt_input_path = tgt_input_path