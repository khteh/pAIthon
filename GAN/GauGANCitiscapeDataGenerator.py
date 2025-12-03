import numpy, h5py, os, tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Normalization, Resizing
from tensorflow.keras.utils import Sequence

class GauGANCityscapeDataGenerator(Sequence):
    '''
    CityscapesDataset Class
    Values:
        paths: (a list of) paths to construct dataset from, a list or string
        img_size: tuple containing the (height, width) for resizing, a tuple
        n_classes: the number of object classes, a scalar
    '''

    def __init__(self, paths, img_size=(256, 512), n_classes=35):
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
        assert all(len(example) == 2 for example in self.examples)

        # Initialize transforms for the real color image
        self.img_transforms = Sequential([
            Resizing(img_size),
            Lambda(lambda img: numpy.array(img)),
            #transforms.ToTensor(),
            Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Initialize transforms for semantic label maps
        self.map_transforms = Sequential([
            Resizing(img_size),
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

        for root, _, files in os.walk(abs_path):
            for f in files:
                if f.endswith(img_suffix):
                    prefix = f[:-len(img_suffix)]
                    attr = 'orig_img'
                elif f.endswith(label_suffix):
                    prefix = f[:-len(label_suffix)]
                    attr = 'label_map'
                else:
                    continue

                if prefix not in self.examples.keys():
                    self.examples[prefix] = {}
                self.examples[prefix][attr] = root + '/' + f

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Load image and maps
        img = Image.open(example['orig_img']).convert('RGB') # color image: (3, h, w)
        label = Image.open(example['label_map'])             # semantic label map: (1, h, w)

        # Apply corresponding transforms
        img = self.img_transforms(img)
        label = self.map_transforms(label).long() * 255

        # Convert labels to one-hot vectors
        label = tf(label, num_classes=self.n_classes)
        label = label.squeeze(0).permute(2, 0, 1).to(img.dtype)
        return (img, label)

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = [], []
        for (x, l) in batch:
            imgs.append(x)
            labels.append(l)
        return tf.stack(imgs, axis=0), tf.stack(labels, axis=0)