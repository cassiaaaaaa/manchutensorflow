import tensorflow as tf
import numpy as np
import collections
import os
import PIL

SAVE_PATH = '.../manchu.tfrecords'#待补充
PNT_PATH_TRAIN = '.../pnt/'#存放pnt文件的文件夹，pnt文件夹分为train,test
PNT_PATH_TEST = '.../pnt/'#存放pnt文件的文件夹，pnt文件夹分为train,test
#PNTcombine_PATH = '.../combinebig.pnt'#存放合并后的大pnt
PNTfiles_Train = os.listdir(PNT_PATH_TRAIN)
PNTfiles_Test = os.listdir(PNT_PATH_TEST)

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
Dataset = collections.namedtuple('Dataset', ['data', 'target'])
                                  
labels_num = 1193

FLAGS = None
#需要定义一个validation_size，或者先shuffle再截取三分之一

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#把修改好的pnt转换成tfrecords。这里是转换成一个tfrecords，如果想转换成多个也可以写for ii in range(n):..每个图都做成一个tfrecords?
def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples
  
  if images.shape[0] != num_examples:
  raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

#再加一个数据预处理，把pnt文件转化成一个一个的image,扩展维度，每个image包括长宽等信息。。。
#即全部pnt是一个文件分为三部分，数量，image和label.image再细分。
#当前输入：存储pnt的文件夹
#输出：一个大文件，有三个部分，数量，image和label.image
#初始pnt文件夹中的pnt格式应该为：unsigned short usSmplen, unsigned short usCode[30], unsigned char ucWid, unsigned char usHei,...
#...unsigned char *pbit，只有二值图没有灰度图。
#要不要把宽度归一化? resize_image_with_crop_or_pad

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

 #从原始pnt中读入，并修改大小，进行宽度归一化
  train_images,train_labels = extract_pnt(PNT_PATH_TRAIN, one_hot=one_hot)
  test_images,test_labels = extract_pnt(PNT_PATH_TEST, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  
  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  #在DataSet类中完成shuffle等操作
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)
  
  return Datasets(train=train, validation=validation, test=test)

#把pnt拆成所需要的格式
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)

def _read16(bytestream):
  dt = numpy.dtype(numpy.uint16).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(2), dtype=dt)

def _read16_30(bytestream):
  dt = numpy.dtype(numpy.uint16).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(60), dtype=dt)

def _read8(bytestream):
  dt = numpy.dtype(numpy.uint8).newbyteorder('>')
  return numpy.fromfile(bytestream.read(1), dtype=dt)

def dense_to_one_hot(labels_dense, num_classes):#把labels转化成onehot格式
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_pnt(PNT_PATH, one_hot=False, num_classes=labels_num):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  import Image
  
  PNTfiles = os.listdir(PNT_PATH)
  SmpLen = 0
  SmpNum = 0
  for file in PNTfiles:
      if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
         with open(path+"/"+file,'rb') as f:
          while(f.read(next)!=''):        
            TmpLen = _read16(f)
            labels = _read16_30(f)
            cols = _read8(f)
            rows = _read8(f)
            buf = f.read(rows * cols)
            data = numpy.frombuffer(buf, dtype=numpy.uint8)
            images[SmpNum] = data.reshape(cols, rows, 1)
            
            #图像压缩,宽归一化为100
            cols = 60
            rows = images[SmpNum].shape[0]*60/images[SmpNum].shape[1]
            images[SmpNum]=images[SmpNum].resize(cols,rows)
            SmpLen = SmpLen + TmpLen
            TmpNum = TmpNum + 1
            if one_hot:
              return dense_to_one_hot(labels, num_classes)
            return images,labels
    
            
class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].灰度归一化是要加的
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]
  
#主函数
def main(unused_argv):
  # Get the data.
  data_sets = read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/tmp/data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
