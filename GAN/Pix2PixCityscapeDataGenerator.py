import numpy, h5py, os, tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Normalization
from tensorflow.keras.utils import Sequence

def scale_width(img, target_width, method):
    '''
    Function that scales an image to target_width while retaining aspect ratio.
    '''
    w, h = img.size
    if w == target_width: return img
    target_height = target_width * h // w
    return img.resize((target_width, target_height), method)

class Pix2PixCityscapeDataGenerator(Sequence):
    '''
    CityscapesDataset Class
    Values:
        paths: (a list of) paths to load examples from, a list or string
        target_width: the size of image widths for resizing, a scalar
        n_classes: the number of object classes, a scalar
    '''
    def __init__(self, paths, target_width=1024, n_classes=35):
        super().__init__()

        self.n_classes = n_classes

        # Collect list of examples
        self.examples = {}
        if type(paths) == str:
            self.load_examples_from_dir(paths)
        elif type(paths) == list:
            for path in paths:
                self.load_examples_from_dir(path)
        else:
            raise ValueError('`paths` should be a single path or list of paths')

        self.examples = list(self.examples.values())
        assert all(len(example) == 3 for example in self.examples)

        # Initialize transforms for the real color image
        self.img_transforms = Sequential([
            Lambda(lambda img: scale_width(img, target_width, Image.BICUBIC)),
            Lambda(lambda img: numpy.array(img)),
            #transforms.ToTensor(),
            Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Initialize transforms for semantic label and instance maps
        self.map_transforms = Sequential([
            Lambda(lambda img: scale_width(img, target_width, Image.NEAREST)),
            Lambda(lambda img: numpy.array(img)),
            #transforms.ToTensor(),
        ])

    def load_examples_from_dir(self, abs_path):
        '''
        Given a folder of examples, this function returns a list of paired examples.
        '''
        assert os.path.isdir(abs_path)

        img_suffix = '_leftImg8bit.png'
        label_suffix = '_gtFine_labelIds.png'
        inst_suffix = '_gtFine_instanceIds.png'

        for root, _, files in os.walk(abs_path):
            for f in files:
                if f.endswith(img_suffix):
                    prefix = f[:-len(img_suffix)]
                    attr = 'orig_img'
                elif f.endswith(label_suffix):
                    prefix = f[:-len(label_suffix)]
                    attr = 'label_map'
                elif f.endswith(inst_suffix):
                    prefix = f[:-len(inst_suffix)]
                    attr = 'inst_map'
                else:
                    continue

                if prefix not in self.examples.keys():
                    self.examples[prefix] = {}
                self.examples[prefix][attr] = root + '/' + f

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Load image and maps
        img = Image.open(example['orig_img']).convert('RGB') # color image: (3, 512, 1024)
        inst = Image.open(example['inst_map'])               # instance map: (512, 1024)
        label = Image.open(example['label_map'])             # semantic label map: (512, 1024)

        # Apply corresponding transforms
        img = self.img_transforms(img)
        inst = self.map_transforms(inst)
        label = self.map_transforms(label).long() * 255

        # Convert labels to one-hot vectors
        label = tf.zeros(self.n_classes, img.shape[1], img.shape[2]).scatter_(0, label, 1.0).to(img.dtype)

        # Convert instance map to instance boundary map
        #bound = torch.ByteTensor(inst.shape).zero_()
        bound = tf.zeros_like(inst.shape, dtype=tf.uint8)
        bound[:, :, 1:] = bound[:, :, 1:] | (inst[:, :, 1:] != inst[:, :, :-1])
        bound[:, :, :-1] = bound[:, :, :-1] | (inst[:, :, 1:] != inst[:, :, :-1])
        bound[:, 1:, :] = bound[:, 1:, :] | (inst[:, 1:, :] != inst[:, :-1, :])
        bound[:, :-1, :] = bound[:, :-1, :] | (inst[:, 1:, :] != inst[:, :-1, :])
        bound = bound.to(img.dtype)
        return (img, label, inst, bound)

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_fn(batch):
        imgs, labels, insts, bounds = [], [], [], []
        for (x, l, i, b) in batch:
            imgs.append(x)
            labels.append(l)
            insts.append(i)
            bounds.append(b)
        return (
            tf.stack(imgs, axis=0),
            tf.stack(labels, axis=0),
            tf.stack(insts, axis=0),
            tf.stack(bounds, axis=0),
        )    
