import glob
import numpy as np
import random
import imageio
from scipy.ndimage import rotate
from skimage.transform import resize
from torch.utils.data import Dataset
import os

class CPCDataset(Dataset):
    def __init__(self, is_train, split_id, enable_aug=True):
        self.enable_aug = enable_aug
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        split_dir = os.path.join(current_dir, '..', 'split')
        
        if is_train == True:
            split_file = os.path.join(split_dir, f'train_wht_pub{split_id}.txt')
        else:
            split_file = os.path.join(split_dir, f'val_wht_pub{split_id}.txt')
            
        self.wli_list = open(split_file).readlines()
        self.is_train = is_train
        self.wli_list = list(map(lambda x: x.strip(), self.wli_list))
        self.wli_label = list(map(lambda x: 'hyperplastic_lesions' not in x, self.wli_list))
        self.wli_label = np.array(self.wli_label, dtype=np.int8)
        unique, counts = np.unique(self.wli_label, return_counts=True)
        print('wli:', dict(zip(unique, counts)))

    def __getitem__(self, index):
        wli_path = self.wli_list[index]
        nbi_paths = glob.glob(wli_path.replace('White_light', 'NBI'))
        if len(nbi_paths) == 0:
            raise Exception("%s NBI is empty" % wli_path)
        nbi_path = random.choice(nbi_paths)
        wli_img = imageio.imread(wli_path)
        nbi_img = imageio.imread(nbi_path)
        # augmentation
        if self.is_train and self.enable_aug:
            angle = random.randint(-180, 180)
            wli_img = rotate(wli_img, angle)
            nbi_img = rotate(nbi_img, angle)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 0)
                wli_img = np.flip(wli_img, 0)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 1)
                wli_img = np.flip(wli_img, 1)

        wli_img = resize(wli_img, (448, 448))
        nbi_img = resize(nbi_img, (448, 448))
        wli_img = np.swapaxes(wli_img, 0,-1)
        nbi_img = np.swapaxes(nbi_img, 0,-1)

        label = self.wli_label[index]
        return wli_img, nbi_img, label, wli_path,nbi_path

    def __len__(self):
        return len(self.wli_list)


class MultiClassPairDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 enable_aug=True,
                 target_size=448,
                 wli_dirname='WLI',
                 nbi_dirname='NBI',
                 class_map=None,
                 image_exts=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        self.root_dir = root_dir
        self.split = split
        self.enable_aug = enable_aug
        self.target_size = target_size
        self.wli_dirname = wli_dirname
        self.nbi_dirname = nbi_dirname
        self.image_exts = image_exts
        self.is_train = split == 'train'

        if class_map is None:
            class_map = {
                '正常': 0,
                '低瘤': 1,
                '高瘤': 2,
                '鳞状细胞癌': 3,
            }
        self.class_map = class_map

        self.samples = []
        label_stats = {v: 0 for v in class_map.values()}

        split_dir = os.path.join(root_dir, split)
        for class_name, label in class_map.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for patient_name in os.listdir(class_dir):
                patient_dir = os.path.join(class_dir, patient_name)
                if not os.path.isdir(patient_dir):
                    continue
                wli_dir = os.path.join(patient_dir, wli_dirname)
                nbi_dir = os.path.join(patient_dir, nbi_dirname)
                if not os.path.isdir(wli_dir) or not os.path.isdir(nbi_dir):
                    continue

                wli_files = self._collect_images(wli_dir)
                nbi_files = self._collect_images(nbi_dir)
                if len(wli_files) == 0 or len(nbi_files) == 0:
                    continue

                nbi_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in nbi_files}
                for wli_path in wli_files:
                    stem = os.path.splitext(os.path.basename(wli_path))[0]
                    nbi_path = nbi_by_stem.get(stem)
                    if nbi_path is None:
                        nbi_path = random.choice(nbi_files)
                    self.samples.append((wli_path, nbi_path, label))
                    label_stats[label] = label_stats.get(label, 0) + 1

        print('multiclass:', label_stats)

    def _collect_images(self, folder):
        files = []
        for name in os.listdir(folder):
            if name.lower().endswith(self.image_exts):
                files.append(os.path.join(folder, name))
        files.sort()
        return files

    def _ensure_3ch(self, img):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        return img

    def __getitem__(self, index):
        wli_path, nbi_path, label = self.samples[index]
        wli_img = imageio.imread(wli_path)
        nbi_img = imageio.imread(nbi_path)
        wli_img = self._ensure_3ch(wli_img)
        nbi_img = self._ensure_3ch(nbi_img)

        if self.is_train and self.enable_aug:
            angle = random.randint(-180, 180)
            wli_img = rotate(wli_img, angle)
            nbi_img = rotate(nbi_img, angle)

            if random.random() > 0.5:
                wli_img = np.flip(wli_img, 0)
                nbi_img = np.flip(nbi_img, 0)

            if random.random() > 0.5:
                wli_img = np.flip(wli_img, 1)
                nbi_img = np.flip(nbi_img, 1)

        if self.target_size is not None:
            if isinstance(self.target_size, int):
                size = (self.target_size, self.target_size)
            else:
                size = self.target_size
            wli_img = resize(wli_img, size)
            nbi_img = resize(nbi_img, size)

        wli_img = np.swapaxes(wli_img, 0, -1)
        nbi_img = np.swapaxes(nbi_img, 0, -1)
        return wli_img, nbi_img, label, wli_path, nbi_path

    def __len__(self):
        return len(self.samples)
