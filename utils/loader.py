from pathlib import Path

import tensorflow as tf
from tensorflow.data import Dataset


@tf.function
def preprocess_image(image, width=224, height=224):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [width, height])
    image /= 255.0  # normalize to [0,1] range
    return image


@tf.function
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


class Bunch:
    def __init__(self, dirname):
        self.root = Path(dirname)

    @property
    def class_names(self):
        return {file.name: k
                for k, file in enumerate(self.root.iterdir())}

    def __call__(self, match='*.jpg'):
        return {file.as_posix(): self.class_names[file.parent.name]
                for file in self.root.rglob(match)}


class PathDataset:
    def __init__(self, data_dir, match='*.jpg'):
        self.bunch = Bunch(data_dir)
        self.paths = Dataset.from_tensor_slices(list(self.bunch(match).keys()))
        self.labels = Dataset.from_tensor_slices(
            list(self.bunch(match).values()))

    @property
    def images(self):
        return self.paths.map(load_and_preprocess_image)

    @property
    def dataset(self):
        return Dataset.zip((self.images, self.labels))

    def __len__(self):
        return tf.data.Dataset.cardinality(self.paths)

    def __call__(self, batch_size, shuffle=False):
        # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
        # 被充分打乱。
        if shuffle:
            n = len(self)
            ds = self.dataset.shuffle(buffer_size=n)
        else:
            ds = self.dataset
        # ds = ds.repeat()
        ds = ds.batch(batch_size)
        # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds
