# -*- coding: utf-8 -*-
import socket
import os
import numpy as np
import sys
import argparse
import glob
import time
from PIL import Image
from StringIO import StringIO
import caffe
import stat

def resize_image(data, sz=(256, 256)):
    """
    Resize image. Please use this resize logic for best results instead of the 
    caffe, since it was used to generate training dataset 
    :param str data:
        The image data
    :param sz tuple:
        The resized image dimensions
    :returns bytearray:
        A byte array with the resized image
    """
    img_data = str(data)
    #print(data)
    im = Image.open(StringIO(img_data))
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize(sz, resample=Image.BILINEAR)
    fh_im = StringIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)
    return bytearray(fh_im.read())

def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
    output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.
    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
        A Caffe network with which to process pimg afrer preprocessing.
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        img_data_rs = resize_image(pimg, sz=(256, 256))
        image = caffe.io.load_image(StringIO(img_data_rs))

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = max((H - h) / 2, 0)
        w_off = max((W - w) / 2, 0)
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                    **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []

if __name__ == '__main__':
    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf-8')
    sk=socket.socket()
    print(sk)
    address=('192.168.1.196',6667)
    sk.bind(address)     
    sk.listen(5)          
    print('waiting........ ')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))     
    while 1:
        conn,addr = sk.accept()
        while 1:
            # client = conn
            #data = conn.recv(1024)                  
            clientfile = conn.makefile('rw',0)
            # client.send("HTTP/1.1 200 OK\r\n\r\n".encode("utf8"))
            # client.send("<h1 style='color:red'>Hello,yuan</h1>".encode("utf8"))
            #print('data',data)
            #filename,filesize=data.encode('utf-8','ignore').split('|')   
	    filename,filesize = clientfile.readline().strip().split('|')
            print('filename',filename)
            #filesize = os.stat(data)
            path = os.path.join(BASE_DIR,'im',filename)
            filesize=int(filesize)

            #os.chmod(path,stat.S_IXGRP)
            #os.chmod(path,stat.S_IWOTH)

            f = open(path,'ab')
            has_receive=0
            while has_receive!=filesize:
                data=conn.recv(1024)                       
                f.write(data)
                has_receive+=len(data)

            # by zhangzheng
            f.close()
            model_def = 'nsfw_model/deploy.prototxt'
            pretrained_model = 'solver_his/solver_iter_30000.caffemodel'
            nsfw_net = caffe.Net(model_def,  # pylint: disable=invalid-name
                    pretrained_model, caffe.TEST)
            #nsfw_net = caffe.Net(args.model_def,  # pylint: disable=invalid-name
            #	args.pretrained_model, caffe.TEST)

            # Load transformer
            # Note that the parameters are hard-coded for best results
            caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
            caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
            caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
            caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
            caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

            # Classify.
            #image_data=data
            image_data = open(path).read()
            
            scores = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=['prob'])
     
            # by zhangzheng

            print('done')
            print(path)
            print(str(scores[1]))
            conn.send(str(scores[1]))
            
            if has_receive>=filesize:
                break
