
import os
from typing import Callable, Optional, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _list_images(folder: str) -> List[str]:
    files = []
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    for name in os.listdir(folder):
        base, ext = os.path.splitext(name)
        if ext.lower() in IMG_EXTS:
            files.append(name)
    return sorted(files)

class PairedFolderDataset(Dataset):
    """
    Pairs images from two folders by identical filename.
    Example:
      root_defect = data/train/defect
      root_clean  = data/train/defect_free
    """
    def __init__(
        self,
        root_defect: str,
        root_clean: str,
        transform: Optional[Callable] = None,
        to_tensor_first: bool = True,
    ):
        self.root_defect = root_defect
        self.root_clean = root_clean
        defect_files = _list_images(root_defect)
        clean_files  = _list_images(root_clean)
        # Use intersection so missing files are ignored
        defect_set = set(defect_files)
        clean_set = set(clean_files)
        common = sorted(defect_set & clean_set)
        if len(common) == 0:
            raise RuntimeError("No matching filenames between "
                               f"{root_defect} and {root_clean}. "
                               "Ensure both folders contain the same image names.")
        self.files = common
        self.transform = transform
        self.to_tensor_first = to_tensor_first

        # Default transform: ToTensor only (values in [0,1])
        if self.transform is None:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        path_defect = os.path.join(self.root_defect, name)
        path_clean  = os.path.join(self.root_clean, name)
        img_defect = Image.open(path_defect).convert("RGB")
        img_clean  = Image.open(path_clean).convert("RGB")
        x = self.transform(img_defect)
        y = self.transform(img_clean)
        return x, y, name
