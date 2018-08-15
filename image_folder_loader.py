import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    camids = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for filename in os.listdir(d):
            if is_image_file(filename):
                path = '{0}/{1}'.format(target, filename)
                item = (path, class_to_idx[target])
                images.append(item)
                cameras = filename.split("c")
                camids.append(int(cameras[1][0]))

    return images, camids


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolderLoader(data.Dataset):
    def __init__(self, root, transform_1=None,
                 transform_2=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs, camids = make_dataset(root, class_to_idx)
        self.root = root
        self.imgs = imgs
        self.camids = camids
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        camid = self.camids[index]
        img = self.loader(os.path.join(self.root, path))
        if self.transform_1 is not None:
            img1 = self.transform_1(img)
        if self.transform_2 is not None:
            img2 = self.transform_2(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target, camid

    def __len__(self):
        return len(self.imgs)