from torch.utils.data import Dataset
from skimage.io import imread
from glob import glob
import numpy as np

class FacadesDataset(Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.file_list = sorted(glob(root_dir + '/' + mode + '/**.jpg'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = imread(self.file_list[idx])
        _, w, _ = img.shape
        w2 = w // 2
        moved = np.moveaxis(img, 2, 0)
        left = moved[:, :, :w2]
        right = moved[:, :, w2:]
        # Первый элемент кортежа -- разметка,
        # второй -- реальная картинка
        return right, left
