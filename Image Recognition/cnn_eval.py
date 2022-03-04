import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter

def decode_image(filename, image_type, resize_shape, channels=0):
  
    value = tf.read_file(filename)

    # Decode image
    if image_type == 'png':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    if resize_shape is not None and image_type in ['png', 'jpeg']:
        decoded_image = tf.image.resize_images(decoded_image, resize_shape)
    return decoded_image

# Return a dataset created from the image file paths
def get_dataset(image_paths, image_type, resize_shape, channels):
    filename_tensor = tf.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)
    
    # mapping function converts raw contents from images into 
    # usable pixel data
    def _map_fn(filename):
        return ref_decode_image(filename, 
                                image_type,
                                resize_shape, 
                                channels=channels)
    return dataset.map(_map_fn)
  
# Get the decoded image data from the input image file paths
def get_image_data(image_paths, image_type=None, resize_shape=None, channels=0):
    dataset = get_dataset(image_paths, image_type, resize_shape, channels)
    iterator =tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_image = iterator.get_next()
    # CODE HERE
    image_data_list = []
    with tf.compat.v1.Session() as sess:
        for i in range(len(image_paths)):
            image_data = sess.run(next_image)
            image_data_list.append(image_data)
    return image_data_list

# Load and resize an image using PIL, and return its pixel data
def pil_resize_image(image_path, resize_shape,
    image_mode='RGBA', image_filter=None):
    im = Image.open(image_path)
    converted_im = im.convert(image_mode)
    resized_im = converted_im.resize(resize_shape, Image.LANCZOS)
    if image_filter is not None:
        resized_im = resized_im.filter(image_filter)
    return np.asarray(resized_im.getdata())


class MNISTModel(object):
    # Model Initialization
    def __init__(self, input_dim, output_size):
        self.input_dim = input_dim
        self.output_size = output_size
    
    # Get logits from the dropout layer
    def get_logits(self, dropout):
        logits = tf.layers.dense(dropout,self.output_size,name='logits')
        return logits
    
    # CNN Layers
    def model_layers(self, inputs, is_training):
        
        reshaped_inputs = tf.reshape(inputs, [-1, self.input_dim, self.input_dim, 1])
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=reshaped_inputs,filters=32,kernel_size=[5,5],padding='same',activation=tf.nn.relu,name='conv1')
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2,name='pool1')
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5,5],padding='same',activation=tf.nn.relu,name='conv2')
        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2,name='pool2')
        
        # Dense Layer
        hwc = pool2.shape.as_list()[1:]
        flattened_size = hwc[0] * hwc[1] * hwc[2]
        pool2_flat = tf.reshape(pool2, [-1, flattened_size])
        dense = tf.layers.dense(pool2_flat, 1024,
            activation=tf.nn.relu, name='dense')
        # Apply Dropout
        dropout = tf.layers.dropout(dense, rate=0.4,
            training=is_training)
        # Get and Return Logits
        return self.get_logits(dropout)
      
    def run_model_setup(self, inputs, labels, is_training):

        logits = self.model_layers(inputs, is_training)

        # convert logits to probabilities with softmax activation
        self.probs = tf.nn.softmax(logits, name='probs')
        # round probabilities
        self.predictions = tf.math.argmax(
            self.probs, axis=-1, name='predictions')
        class_labels = tf.math.argmax(labels, axis=-1)
        # find which predictions were correct
        is_correct = tf.math.equal(
            self.predictions, class_labels)
        is_correct_float = tf.cast(
            is_correct,
            tf.float32)

        # compute ratio of correct to incorrect predictions
        self.accuracy = tf.math.reduce_mean(
            is_correct_float)

        # train model
        if self.is_training:
            labels_float = tf.cast(
                labels, tf.float32)
            # compute the loss using cross_entropy
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_float,
                logits=logits)
            self.loss = tf.math.reduce_mean(
                cross_entropy)
            # use adam to train model
            adam = tf.compat.v1.train.AdamOptimizer()
            self.train_op = adam.minimize(
                self.loss, global_step=self.global_step)
