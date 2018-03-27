import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet
from csv import reader
import re
from skimage.transform import rescale


# read data from file    
def load_file(filename):
    
    val = [];
    with open(filename) as afile:
        r = reader(afile)
        for line in r:
            val.append(line[0]);
            
    return val;

	
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]
	


nb_classes = 43

xFeed = tf.placeholder(tf.float32, (None, 227, 227, 3))
resized = tf.image.resize_images(xFeed, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the specifically for 
# traffic signs model.
fc7 = AlexNet(resized, feature_extract=True)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

print(fc7);

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images

imList = load_file("PascalSentences/FilesList.txt");
imList = ["PascalSentences/"+x.split('|')[0] for x in imList]
print(imList[0]);

images = [];

for im in imList:
	xi = imread(im).astype(np.float32);
	xi = rescale(xi/255.0, ( 227.0/float(min(xi.shape[0],xi.shape[1])) ));
	xi = crop_center(xi*255.0,227,227);
	xi = xi-np.mean(xi);
	images.append(xi);
	
print("****************\nImages Loaded\n*****************")

"""	
images = [ imread(x).astype(np.float32) for x in imList ];
images = [ rescale(x/255.0, ( 227.0/float(min(x.shape[0],x.shape[1])) )) for x in images]; 
images = [crop_center(x*255.0,227,227) for x in images];
images = [ x-np.mean(x) for x in images ];
"""

"""
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)
"""

# Run Inference
t = time.time()

output = sess.run(fc7, feed_dict={xFeed: images[0:500]})
print((output));

output = sess.run(fc7, feed_dict={xFeed: images[10:20]})
print((output));



f = open("PascalAlexnetFeatures.txt","w");

for x in output:
	f.write(" ".join(map(str,x))+"\n");


"""
# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
"""