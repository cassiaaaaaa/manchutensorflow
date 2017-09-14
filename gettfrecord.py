import numpy as np
import tensorflow as tf
import collections 
import os
import PIL
from PIL import Image
import struct
from bitstring import BitArray
from functools import reduce 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

labelset = "abcdefghijklmnopqrstuvwxyz,."

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _read16(bytestream):
    dt = numpy.dtype(numpy.uint16).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(2), dtype=dt)

def _read8(bytestream):
    dt = numpy.dtype(numpy.uint8).newbyteorder('>')
    return numpy.fromfile(bytestream.read(1), dtype=dt)

def write_image_annotation_pairs_to_tfrecord(img_path, tfrecords_filename):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for root,subfolder,fileList in os.walk(img_path):
        for fname in fileList:
            fname = os.path.join(root,fname)
            flen = os.path.getsize(fname)
            print(fname)
            
            with open(fname,'rb') as f:
                buf = f.read()
                f.seek(0)
                index = 0
                count_num_total = 0
                
                while(index < flen):  
                #while(count_num_total < 1714):
                    pnt_len,width,height = struct.unpack_from('<HHH',buf,index)   
                    index = index + 6
                    
                    count_num_total = count_num_total + 1 
                    data_len = ((width+7)//8*height)
                    pre_len = data_len + 6
                    label_len = pnt_len-6-(width+7)//8*height
                    
                    
                    data_location_str = 'B'*data_len
                    
                    data_raw = struct.unpack_from(data_location_str,buf,index)
                    #print(data_raw)
                    
                    #data_raw = str(data_raw)
                    #data_raw = [ord(b) for b in data_raw]
                    data_raw = [list(bin(b)[2:].rjust(8,'0')) for b in data_raw]
                    #print(data_raw)
                    
                    data_list = [int(item)*255 for sublist in data_raw for item in sublist]
                    #print(data_list)
                    size = len(data_list)
                    print(size)
                    data_list = data_list[:width*height]
                    
                    data_list = np.array(data_list)
                    print(type(data_list),data_list)
                    data_mat = data_list.reshape(width,height)
                    print(width*height,width,height)
                    #data_mat2 = np.reshape(data_list,(width, height))
                    #data_mat2 = Image.fromarray(data_mat2)
                    #print(data_mat2)
                    #plt.subplot(2,1,2)
                    plt.imshow(data_mat, cmap=cm.gray) # 指明 cmap=cm.gray
                    plt.show()
                    #plt.imshow(data_mat)
                    #show_images(data_mat)

                    #data_raw_list = [];
                    #data_raw_list = [data_raw_list.extend(bin(b)[2:].rjust(8,'0')) for b in data_raw]
                    #data_b = [struct.pack(data_b,b) for b in data_raw]
                    #reduce(lambda x,y: x.extend(y),data_raw)
                    #data_int = [int(c) for c in (for d in data_raw) ]

                    
                    #data_raw_r = reduce(data_raw)
                    #print(data_raw_r)
                    
                    ss
                    test1 = BitArray(data_raw[1])
                    print(data_raw[1])
                    
                    print(type(test1),list(test1.bin[2:]))
                   
                    data_raw_l = list(data_raw[:])
                    data_raw_encode = ''
                    data_raw_encode = [data_raw_encode + data_raw_l[i] for i in range(data_len)]
                    print(type(data_raw_encode),data_raw_encode)
                    
                     
                    f.read(2) 
                    f.read(2) ,
                    f.read(2)  
                    
                    f.seek(index)
                    data_raw = f.read(data_len)
                    data_raw_d = str(data_raw)
                    print(type(data_raw),type(data_raw_d))
                    print(data_raw_d)
                    data_raw_bit = BitArray(data_raw_d)
                    print(type(data_raw_bit),data_raw_bit)

                    label_raw = f.read(label_len)
                    index = index + pnt_len
                    label_raw_d = label_raw.decode()
                    
                    
                                        
                    if label_raw_d == 'I': 
                        label_raw_d = "i"
                        
                    label_index = [labelset.index(c) for c in list(label_raw_d) if c in labelset]

                    context = tf.train.Features(feature={
                        'height': _int64_feature(height),                                
                        'width': _int64_feature(width),
                        'label_len': _int64_feature(label_len),
                        'data_raw': _bytes_feature(data_raw)
                        })
                    featureLists = tf.train.FeatureLists(feature_list={
                        'label_index': _int64_feature_list(label_index)
                        })
                    sequence_example = tf.train.SequenceExample(
                        context=context,feature_lists =featureLists
                        )
                    writer.write(sequence_example.SerializeToString())
                    #writer.close()#pre\n",
                
        writer.close()
                
                   

def write_test(img_path ,tfrecords_filename = None):
    write_image_annotation_pairs_to_tfrecord(img_path=img_path,tfrecords_filename=tfrecords_filename)
    
def show_images(images):
    n = images.shape[0]
    _,figs = plt.subplots(1,n,figsize = (15, 15))
    for i in range(n):
        figs[i].imshow(np.array(images[i]))
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

if __name__=='__main__':
    write_test(img_path='/home/kikyo/文档/manchu-tensorflow/data/raw_data_combine/test',tfrecords_filename='/home/kikyo/文档/manchu-tensorflow/data/tfrecords/temp.tfrecords')

