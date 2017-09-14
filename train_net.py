import os
import os.path as osp
import argparse
import sys
import numpy as np
from easydict import EasyDict as edict
import tensorflow as tf
from time import strftime, localtime

-

def get_encode_decode_dict():
    encode_maps = {}
    decode_maps = {}
    for i, char in enumerate(__C.CHARSET, 1):
        encode_maps[char] = i
        decode_maps[i] = char
    encode_maps[__C.SPACE_TOKEN] = __C.SPACE_INDEX
    decode_maps[__C.SPACE_INDEX] = __C.SPACE_TOKEN
    return encode_maps,decode_maps

def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue):
    """Return image/annotation tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/annotation tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    
    Returns
    -------
    image, annotation : tuple of tf.int32 (image, annotation)
        Tuple of image/annotation tensors
    """
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features,sequence_features = tf.parse_single_sequence_example( serialized_example,
        context_features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'time_step': tf.FixedLenFeature([], tf.int64),
            'label_len': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string), },
        sequence_features={
            'label': tf.FixedLenSequenceFeature([], tf.int64),})
    
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label_len = tf.cast(features['label_len'], tf.int32)
    label = tf.cast(sequence_features['label'],tf.int32)
    label = tf.reshape(label,[cfg.MAX_CHAR_LEN])
    #image_shape = tf.pack([height, width, 3])
    image_shape = tf.parallel_stack([height, width, 3])
    image = tf.reshape(image,image_shape)

    img_size = cfg.IMG_SHAPE #160,60
    time_step = tf.constant(cfg.TIME_STEP,tf.int32)

    if cfg.NCHANNELS==1: image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,size=(img_size[1],img_size[0]),method=tf.image.ResizeMethod.BILINEAR)
    image = tf.transpose(image,perm=[1,0,2])
    image = tf.cast(tf.reshape(image,[img_size[0],cfg.NUM_FEATURES]),dtype=tf.float32)/255.

    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third
    # dimension
    #annotation_shape = tf.pack([height, width, 1])
    # image = tf.reshape(image, image_shape)

    return image, label,label_len,time_step

def accuracy_calculation(original_seq,decoded_seq,ignore_value=0,isPrint = True):
    if  len(original_seq)!=len(decoded_seq):
        print('original lengths is different from the decoded_seq,please check again')
        return 0
    count = 0
    for i,origin_label in enumerate(original_seq):
        decoded_label  = [j for j in decoded_seq[i] if j!=ignore_value]
        org_label = [l for l in origin_label if l!=ignore_value]
        if isPrint and i<cfg.VAL.PRINT_NUM:
            print('seq{0:4d}: origin: {1} decoded:{2}'.format(i,origin_label,decoded_label))
        if org_label == decoded_label: count+=1
    return count*1.0/len(original_seq)

def get_log_dir(imdb):
    log_dir = osp.abspath(\
        osp.join(__C.ROOT_DIR, 'logs', __C.LOG_DIR, imdb.name, strftime("%Y-%m-%d-%H-%M-%S", localtime())))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_output_dir(imdb, weights_filename):
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR))
    if weights_filename is not None:
        outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def incluude_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

-
--

def train_net(network, imgdb, pre_train,output_dir, log_dir, max_iters=40000, restore=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imgdb, pre_train,output_dir, logdir= log_dir)
        print('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print('done solving')

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Train a lstm network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=700000, type=int)
   # parser.add_argument('--cfg', dest='cfg_file',
   #                     help='optional config file',
    #                    default=None, type=str)
   # parser.add_argument('--pre_train', dest='pre_train',
    #                    help='pre trained model',
     #                   default=None, type=str)
    #parser.add_argument('--rand', dest='randomize',
    #                    help='randomize (do not use a fixed seed)',
     #                   action='store_true')
    #parser.add_argument('--network', dest='network_name',
    #                    help='name of the network',
   #                     default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--restore', dest='restore',
                        help='restore or not',
                        default=0, type=int)

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()
    return args
"""

def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'LSTM':
        if name.split('_')[1] == 'train':
            return LSTM_train()
        elif name.split('_')[1] == 'test':
            return LSTM_test()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))

if __name__ == '__main__':
    
    #args = parse_args()
    network_name = "LSTM_train"
    output_network_name = network_name.split('_')[-1]#原来在bash里面的传入参数
    imgdb = edict({'path':'/home/kikyo/文档/manchu-tensorflow/data/tfrecords/traindata.tfrecords','name':'lstm_'+output_network_name,
                   'val_path':'/home/kikyo/文档/manchu-tensorflow/data/tfrecords/validdata.tfrecords' })
 
    output_dir = get_output_dir(imgdb, None)
    log_dir = get_log_dir(imgdb)
    print(('Output will be saved to `{:s}`'.format(output_dir)))
    print(('Logs will be saved to `{:s}`'.format(log_dir)))
    
    network = get_network(args.network_name)
    print(('Use network `{:s}` in training'.format(args.network_name)))
    
    train_net(network, imgdb,
              pre_train=args.pre_train,
              output_dir=output_dir,
              log_dir=log_dir,
              max_iters=args.max_iters,
              restore=bool(int(args.restore)))
    
