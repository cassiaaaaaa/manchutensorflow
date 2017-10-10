import os
import os.path as osp
import getopt
import argparse
import numpy as np
from easydict import EasyDict as edict
import tensorflow as tf
from time import strftime, localtime
import time

DEFAULT_PADDING = 'SAME'
__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# Default GPU device id
__C.GPU_ID = 0
# region proposal network (RPN) or not
__C.POOL_SCALE = 2
__C.MAX_CHAR_LEN = 30
__C.CHARSET = 'abcdefghijklmnopqrstuvwxyz,.'
__C.NCLASSES = len(__C.CHARSET)+2#一个填充的‘’和一个blank_
__C.NCHANNELS = 1
__C.IMG_SHAPE = [960,48]
__C.NUM_FEATURES=__C.IMG_SHAPE[1]

__C.TIME_STEP = __C.IMG_SHAPE[0]//__C.POOL_SCALE

__C.NET_NAME = 'lstm'
__C.TRAIN = edict()
# Adam, Momentum, RMS
__C.TRAIN.SOLVER = 'Adam'
#__C.TRAIN.SOLVER = 'Momentum'
# __C.TRAIN.SOLVER = 'RMS'
# learning rate
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.LEARNING_RATE = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZE = 50000
__C.TRAIN.DISPLAY = 10
__C.TRAIN.LOG_IMAGE_ITERS = 100
__C.TRAIN.NUM_EPOCHS = 2000

__C.TRAIN.NUM_HID = 128
__C.TRAIN.NUM_LAYERS = 2
__C.TRAIN.BATCH_SIZE = 32

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.SNAPSHOT_PREFIX = 'lstm'
__C.TRAIN.SNAPSHOT_INFIX = ''

__C.VAL = edict()
__C.VAL.VAL_STEP = 50#500
__C.VAL.NUM_EPOCHS = 1000
__C.VAL.BATCH_SIZE = 32
__C.VAL.PRINT_NUM = 5

__C.RNG_SEED = 3

#__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.ROOT_DIR = "/media/kikyo/数据仓库/manchu-tensorflow/data/"
__C.TEST = edict()
__C.EXP_DIR = 'default'
__C.LOG_DIR = 'default'

__C.SPACE_INDEX = 0
__C.SPACE_TOKEN = ''

#indices对应
encode_maps = {}
decode_maps = {}

for i,char in enumerate(cfg.CHARSET,1):
    encode_maps[char] = i
    decode_maps[i] = char

encode_maps[''] = 0
decode_maps[0] = ['']

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def get_log_dir(imdb):
    log_dir = osp.abspath(\
        osp.join(__C.ROOT_DIR, 'logs', imdb.name, strftime("%Y-%m-%d-%H-%M-%S", localtime())))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_output_dir(imdb, weights_filename):
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output/'))
    if weights_filename is not None:
        outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'LSTM':
        if name.split('_')[1] == 'train':
            return LSTM_train()
        elif name.split('_')[1] == 'test':
            return LSTM_test()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))      

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
            
def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features,sequence_features = tf.parse_single_sequence_example( serialized_example,
        context_features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            #'time_step': tf.FixedLenFeature([], tf.int64),
            'label_len': tf.FixedLenFeature([], tf.int64),
            'data_raw': tf.FixedLenFeature([], tf.string), },
        sequence_features={
            'aligned_label': tf.FixedLenSequenceFeature([], tf.int64),})
    
    image = tf.decode_raw(features['data_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label_len = tf.cast(features['label_len'], tf.int32)
    
    image_shape = tf.parallel_stack([height, width,1])
    image = tf.reshape(image,image_shape)

    img_size = cfg.IMG_SHAPE #960,48
    time_step = tf.constant(cfg.TIME_STEP,tf.int32)

    #if cfg.NCHANNELS==1: image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,size=(img_size[0],img_size[1]),method=tf.image.ResizeMethod.BILINEAR)
    #image = tf.transpose(image,perm=[1,0,2])
    image = tf.cast(tf.reshape(image,[img_size[0],cfg.NUM_FEATURES,1]),dtype=tf.float32)/255.
    #label = tf.serialize_sparse(sequence_features['aligned_label'])
    label = tf.cast(sequence_features['aligned_label'],tf.int32)###
    label = tf.reshape(label,[cfg.MAX_CHAR_LEN])###
    #label = tf.serialize_sparse(sequence_features['aligned_label'])
    
    #indices = tf.decode_raw(sequence_features['aligned_label'],string)
    
    """
    batch_labels = sparse_tuple_from_label(aligned_label.eval())

    label = tf.SparseTensorValue(indices,values,shape)
    label = tf.convert_to_tensor_or_sparse_tensor(label)
    label = tf.serialize_sparse(sequence_features['aligned_label'] ) # for batching
    label = tf.deserialize_many_sparse(label, tf.int64) # post-batching...
    label = tf.cast(label, tf.int32) # for ctc_loss
    """
    #可以针对不一样长的数据。。notice
    #image_shape = tf.parallel_stack([height, width])
    #image = tf.reshape(image,image_shape)

    #img_size = cfg.IMG_SHAPE 
    #time_step = tf.constant(cfg.TIME_STEP,tf.int32)

    #if cfg.NCHANNELS==1: image = tf.image.rgb_to_grayscale(image)
    ##image = tf.image.resize_images(image,size=(img_size[1],img_size[0]),method=tf.image.ResizeMethod.BILINEAR)
    ##image = tf.transpose(image,perm=[1,0,2])
    ##image = tf.cast(tf.reshape(image,[img_size[0],cfg.NUM_FEATURES]),dtype=tf.float32)/255.
   # image = tf.cast(tf.reshape(image,[img_size[0],cfg.NUM_FEATURES]),dtype=tf.float32)
    
    #print("cast-reshape images:",image)
    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third
    # dimension
    #annotation_shape = tf.pack([height, width, 1])
    # image = tf.reshape(image, image_shape)

    return image, label,label_len,time_step
            
def include_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator
            
@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        
        # Return self for chained calls.
        return self
    return layer_decorated
            
class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path,encoding='latin1').item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print("ignore "+key)
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in list(self.layers.items()))+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)
        #return tf.get_variable(name, shape, initializer=None, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def bi_lstm(self, input, num_hids, num_layers, name,img_shape = None ,trainable=True):
        img,img_len = input[0],input[1]#input是一个list结构，分别代表conv4的输出和time_step_len(960)
        #print("bilstm input",input[0],input[1])

        if img_shape:img =tf.reshape(img,shape = img_shape )
        #print("this is a test to bi")
        with tf.variable_scope(name) as scope:
            #stack = tf.contrib.rnn.MultiRNNCell([cell,cell1] , state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(num_hids/2,state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(num_hids/2,state_is_tuple=True)

            output,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,img,img_len,dtype=tf.float32)
            # output_bw_reverse = tf.reverse_sequence(output[1],img_len,seq_axis=1)
            output = tf.concat(output,axis=2)

            stack_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(num_hids, state_is_tuple=True) for _ in range(num_layers)],
                state_is_tuple=True)
            lstm_out,last_state = tf.nn.dynamic_rnn(stack_cell,output,img_len,dtype=tf.float32)
            shape = tf.shape(img)

            batch_size, time_step = shape[0],shape[1]
            #print(batch_size,time_step)
            lstm_out = tf.reshape(lstm_out,[-1,num_hids])
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            # init_weights = tf.contrib.layers.xavier_initializer()
            # init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            W = self.make_var('weights', [num_hids, cfg.NCLASSES], init_weights, trainable, \
                              regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            b = self.make_var('biases', [cfg.NCLASSES], init_biases, trainable)
            logits = tf.matmul(lstm_out,W)+b
            logits = tf.reshape(logits,[batch_size,-1,cfg.NCLASSES])
            logits = tf.transpose(logits,(1,0,2))
            return logits
    @layer
    def lstm(self, input, num_hids, num_layers, name,img_shape = None ,trainable=True):
        img,img_len = input[0],input[1]
        if img_shape:img =tf.reshape(img,shape = img_shape )
        with tf.variable_scope(name) as scope:
            stack_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(num_hids, state_is_tuple=True) for _ in range(num_layers)],
                state_is_tuple=True)
            lstm_out,last_state = tf.nn.dynamic_rnn(stack_cell,img,img_len,dtype=tf.float32)
            shape = tf.shape(img)
            batch_size, time_step = shape[0],shape[1]
            lstm_out = tf.reshape(lstm_out,[-1,num_hids])
            # init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.001, mode='FAN_AVG', uniform=False)
            # init_weights = tf.contrib.layers.xavier_initializer()
            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            W = self.make_var('weights', [num_hids, cfg.NCLASSES], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            b = self.make_var('biases', [cfg.NCLASSES], init_biases, trainable)
            logits = tf.matmul(lstm_out,W)+b
            logits = tf.reshape(logits,[batch_size,-1,cfg.NCLASSES])
            logits = tf.transpose(logits,(1,0,2))
            return logits

    @layer
    def concat(self, input, axis, name):
        with tf.variable_scope(name) as scope:
            concat = tf.concat(values=input,axis=axis)
        return concat

    @layer
    def conv_single(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
       # if c_i==1: input = tf.expand_dims(input=input,axis=3)
        
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1,s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.contrib.layers.xavier_initializer()
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                #print("conv shape:",conv)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)

                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                #print("conv shape:",conv)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer    

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)


    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1], name=name)

    @layer
    def batch_normalization(self,input,name,relu=True, is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

    @layer
    def scale(self, input, c_in, name):
        with tf.variable_scope(name) as scope:

            alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=True,
                                    regularizer=self.l2_regularizer(0.00001))
            beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0), trainable=True,
                                   regularizer=self.l2_regularizer(0.00001))
            return tf.add(tf.multiply(input, alpha), beta)


    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)


    def build_loss(self):
        time_step_batch = self.get_output('time_step_len')
        logits_batch = self.get_output('logits')
        labels = self.get_output('labels')
        label_len = self.get_output('labels_len')
        print(label_len)

        #ctc_loss = tf.nn.ctc_loss(activations=logits_batch,flat_labels=labels,
        #                                     label_lengths=label_len,input_lengths=time_step_batch)
        ctc_loss = tf.nn.ctc_loss(labels=labels,inputs=logits_batch,
                                               sequence_length=label_len)
        loss = tf.reduce_mean(ctc_loss)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits_batch, time_step_batch, merge_repeated=True)
        dense_decoded = tf.cast(tf.sparse_tensor_to_dense(decoded[0], default_value=0), tf.int32)

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss

        return loss,dense_decoded   
    
class LSTM_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []

        self.data = tf.placeholder(tf.float32, shape=[None,None,None,1], name='data') #N*t_s*features*channels
        #self.labels = tf.placeholder(tf.int32,[None],name='labels')##
        self.labels = tf.sparse_placeholder(tf.int32,name='labels')
        self.time_step_len = tf.placeholder(tf.int32,[None], name='time_step_len')
        self.labels_len = tf.placeholder(tf.int32,[None],name='labels_len')

        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data,'labels':self.labels,
                            'time_step_len':self.time_step_len,
                            'labels_len':self.labels_len})
        self.trainable = trainable
        self.setup()

    def setup(self):
        n_classes = cfg.NCLASSES
        
        (self.feed('data')
         .conv_single(3, 3, 32 ,1, 1, name='conv1')#3,3,32,1,1
         .conv_single(3, 3, 64 ,1, 1, name='conv2')#3,3,64,1,1
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv_single(3, 3, 1 ,1, 1, name='conv4',relu=False))
        (self.feed('conv4','time_step_len')
         # .lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))
         .bi_lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))


class LSTM_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []

        self.data = tf.placeholder(tf.float32, shape=[None, None, cfg.NUM_FEATURES], name='data')
        self.time_step_len = tf.placeholder(tf.int32,[None], name='time_step_len')

        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'time_step_len':self.time_step_len})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
         .conv_single(3, 3, 32 ,1, 1, name='conv1',c_i=cfg.NCHANNELS)
         .conv_single(3, 3, 64 ,1, 1, name='conv2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv_single(3, 3, 1 ,1, 1, name='conv4',relu=False))
        (self.feed('conv4','time_step_len')
         # .lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))
         .bi_lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))

def incluude_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

def sparse_tuple_from_label(sequences, dtype=np.int32):

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape 

def sparse_with_tensor(a):
    with tf.Session() as sess:
        a_t = tf.constant(a)
        idx = tf.where(tf.not_equal(a_t, 0))
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
        sparse = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), a_t.get_shape())
        dense = tf.sparse_tensor_to_dense(sparse)
        b = sess.run(dense)
        label_len = np.zeros(a_t.get_shape()[0])
                
        for i in range(0,len(label_len)):
            for j in range(0,(idx.eval().shape[0])):
                if idx.eval()[j,0] == i:
                    label_len[i]= label_len[i]+1
        return sparse,label_len
    
class SolverWrapper(object):
    def __init__(self, sess, network, imgdb, pre_train,output_dir, logdir):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imgdb = imgdb
        self.pre_train=pre_train
        self.output_dir = output_dir
        print('done')
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.summary.FileWriter(logdir=logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=5)

    def snapshot(self, sess, iter):
        net = self.net
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + '_ctc' + infix +
                        '_iter_{:d}'.format(iter + 1) + '.ckpt')
        
        #filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
         #           '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def get_data(self,path,batch_size,num_epochs):
        filename_queue = tf.train.string_input_producer([path], num_epochs=num_epochs)
        image,label,label_len,time_step= read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)
        #TODO：在这里可以对image进行一下规范化，使用函数tf.image.per_image_standardization(image)，
        #其中参数image是一个3-D的张量，形状为[height, width, channels]
        """
        image_batch, label_batch, label_len_batch,time_step_batch = tf.train.shuffle_batch([image,label,label_len,time_step],
                                                                                           batch_size=batch_size,
                                                                                           capacity=9600,
                                                                                           num_threads=4,
                                                                                          min_after_dequeue=6400)
        """
        image_batch, label_batch, label_len_batch,time_step_batch = tf.train.batch([image,label,label_len,time_step],
                                                                                           batch_size=batch_size,
                                                                                           capacity=9600,
                                                                                           num_threads=4)
        
       
        return image_batch, label_batch, label_len_batch,time_step_batch
                                                                                           
    def mergeLabel(self,labels,ignore = 0):
        label_lst = []
        for l in labels:
            while l[-1] == ignore: l = l[:-1]
            label_lst.extend(l)
        return np.array(label_lst)
    
                                                                                      

    def train_model(self, sess, max_iters, restore=False):
        img_b,lb_b,lb_len_b,t_s_b = self.get_data(self.imgdb.path,batch_size= cfg.TRAIN.BATCH_SIZE,num_epochs=cfg.TRAIN.NUM_EPOCHS)
        val_img_b, val_lb_b, val_lb_len_b,val_t_s_b = self.get_data(self.imgdb.val_path,batch_size=cfg.VAL.BATCH_SIZE,num_epochs=cfg.VAL.NUM_EPOCHS)
        #print(img_b,lb_b,lb_len_b,t_s_b)
        
        loss, dense_decoded = self.net.build_loss()

        #用于tensorboard可视化
        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()

        # optimizer
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        else:
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)
        

        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(loss, global_step=global_step)

        # intialize variables
        local_vars_init_op = tf.local_variables_initializer()
        global_vars_init_op = tf.global_variables_initializer()

        combined_op = tf.group(local_vars_init_op, global_vars_init_op)
        sess.run(combined_op)
        restore_iter = 1

        # resuming a trainer
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, tf.train.latest_checkpoint(self.output_dir))
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print('done')
            except:
                raise Exception('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

        timer = Timer()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        loss_min = 0.02
        
        try:
            while not coord.should_stop():
        
                for iter in range(restore_iter, max_iters):
                    timer.tic()
                    
                    # learning rate
                    if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                        sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))                 

                    # get one batch
                    img_Batch,labels_Batch, label_len_Batch,time_step_Batch = \
                        sess.run([img_b,lb_b,lb_len_b,t_s_b])
                    

                    #print("iter test")#test
                    #print(type(tuple(labels_Batch)))
                    #print(tuple(labels_Batch))
                    #label_Batch = self.mergeLabel(labels_Batch,ignore = 0)
                    #print(label_Batch)
                    #label_Batch = tuple(map(tuple, labels_Batch))
                    #decoded_label = [decode_maps]
                   # print("label_:",decode_maps[labels_Batch])
                    print("label_:",labels_Batch)
                    label_Batch,labels_len_Batch = sparse_with_tensor(labels_Batch)
                    label_Batch = label_Batch.eval()
                    #label_Batch = tf.serialize_sparse(label_Batch)
                    #label_Batch = sparse_tuple_from_label(tuple(label_Batch))
                    print("label_len:",labels_len_Batch)
                    #print(label_Batch.indices)

                    #print("label:",label_Batch,label_Batch.indices,"type:",type(label_Batch.indices),type(label_Batch.values))
                    #ss
                    """
                    print(type(labels_Batch))
                    print(labels_Batch)                                                                       
                    label_Batch = tf.deserialize_many_sparse(labels_Batch, tf.int64) # post-batching...
                    label_Batch = tf.cast(label_Batch, tf.int32) # for ctc_loss
                    """
                    # Subtract the mean pixel value from each pixel
                    feed_dict = {
                        self.net.data:          img_Batch,
                        self.net.labels:        label_Batch,
                        self.net.time_step_len: np.array(time_step_Batch),
                        self.net.labels_len:    np.array(labels_len_Batch),
                        self.net.keep_prob:     0.5
                    }

                    #print("after feed_dict")#test
                    #NOTICE。这个feed_dict才定义了输入进ctc_loss的数据格式。。
                    fetch_list = [loss,summary_op,train_op]
                    ctc_loss,summary_str, _ =  sess.run(fetches=fetch_list, feed_dict=feed_dict)

                    self.writer.add_summary(summary=summary_str, global_step=global_step.eval())
                    _diff_time = timer.toc(average=False)
                    
                    
                    #print("img_batch:",img_Batch.shape(),img_Batch)
                    #print("label_batch:",label_Batch)

                    if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                        print('iter: %d / %d, total loss: %.7f, lr: %.7f'%\
                                (iter, max_iters, ctc_loss ,lr.eval()),end=' ')
                        print('speed: {:.3f}s / iter'.format(_diff_time))
                    
                        
                    if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0 or ctc_loss<loss_min:
                        if(ctc_loss<loss_min):
                            print('loss: ',ctc_loss,end=' ')
                            self.snapshot(sess, 1)
                            loss_min = ctc_loss
                        else: self.snapshot(sess, iter)
                    if (iter+1) % cfg.VAL.VAL_STEP == 0 or loss_min==ctc_loss:
                        val_img_Batch,val_labels_Batch, val_label_len_Batch,val_time_step_Batch = \
                            sess.run([val_img_b,val_lb_b,val_lb_len_b,val_t_s_b])
                        #val_label_Batch = self.mergeLabel(val_labels_Batch,ignore = 0)
                        #val_label_Batch = tuple(map(tuple, val_labels_Batch))
                        #val_label_Batch = sparse_tuple_from_label(val_label_Batch)
                        print("val_label_:",val_labels_Batch)
                        
                        val_label_Batch,val_labels_len_Batch = sparse_with_tensor(val_labels_Batch)
                        val_label_Batch = val_label_Batch.eval()
                        print("valabelen:",val_labels_len_Batch)

                        feed_dict = {
                            self.net.data :          val_img_Batch,
                            self.net.labels :         val_label_Batch,
                            self.net.time_step_len : np.array(val_time_step_Batch),
                            self.net.labels_len :     np.array(val_labels_len_Batch),
                            self.net.keep_prob:      1.0
                        }

                        #fetch_list = [dense_decoded]
                        org = val_labels_Batch
                        res =  sess.run(fetches=dense_decoded, feed_dict=feed_dict)
                        acc = accuracy_calculation(org,res,ignore_value=0)
                        print('accuracy: {:.5f}'.format(acc))

                iter = max_iters-1
                self.snapshot(sess, iter)
                coord.request_stop()
        except tf.errors.OutOfRangeError:
            print('finish')
        finally:
            coord.request_stop()
        coord.join(threads)
    
def train_net(network, imgdb, pre_train,output_dir, log_dir, max_iters=40000, restore=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imgdb, pre_train,output_dir, logdir= log_dir)
        print('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print('done solving')
        
if __name__ == '__main__':
    
    #args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
    np.random.seed(cfg.RNG_SEED)
    
    network_name = "LSTM_train"#could be changed
    output_network_name=network_name.split('_')[-1]#原来在bash里面的传入参数
    imgdb = edict({'path':'/media/kikyo/数据仓库/manchu-tensorflow/data/tfrecords/0925-labellen30/train.tfrecords','name':'lstm_'+output_network_name,
                   'val_path':'/media/kikyo/数据仓库/manchu-tensorflow/data/tfrecords/0925-labellen30/valid.tfrecords' })
 
    device_name = '/gpu:{:d}'.format(cfg.GPU_ID)
    print(device_name)
    
    output_dir = get_output_dir(imgdb, None)
    log_dir = get_log_dir(imgdb)
    print(('Output will be saved to `{:s}`'.format(output_dir)))
    print(('Logs will be saved to `{:s}`'.format(log_dir)))
    
    tf.reset_default_graph()#可能是ipython notebook内部的原因，本来在layer内部出现问题，加了这句就好了
    network = get_network(network_name)
    
    print(('Use network `{:s}` in training'.format(network_name)))
    train_net(network, imgdb,
              pre_train=None,
              output_dir=output_dir,
              log_dir=log_dir,
              max_iters=700000,
              #restore=bool(int(args.restore)))
              restore = 0)
    
  
