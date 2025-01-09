'''
Neural Style Transfer   -   Drawing images in the style of another image
'''

import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import os
from datetime import datetime

import tensorflow as tf


global start_process, epochs, steps_per_epoch, image_to_use, style_to_use, generated_dir, photos_dir, paintings_dir, photos_dict, paintings_dict


# Defines the content and style variables, the functions that are used and the training procedure
def initialize():
    global start_process, epochs, steps_per_epoch, image_to_use, style_to_use, generated_dir, photos_dir, paintings_dir, photos_dict, paintings_dict
    start_process = datetime.now()
    mpl.rcParams['figure.figsize'] = (10, 10)
    mpl.rcParams['axes.grid'] = False
    epochs = 5      # Final output quality increases with epoch count (up to max ~20)
    steps_per_epoch = 50

    root_dir = os.path.dirname(os.path.abspath(__file__))  # os.getcwd()
    photos_dir = root_dir + "/images/content/"
    paintings_dir = root_dir + "/images/styles/"
    generated_dir = root_dir + "/images/generated/"
    image_extension = ".jpg"

    # Setting up Images and Styles
    photos_dict = {}
    paintings_dict = {}
    for filename in os.listdir(photos_dir):
        if filename.endswith(image_extension):
            photos_dict[filename[:-4]] = photos_dir + filename

    for filename in os.listdir(paintings_dir):
        if filename.endswith(image_extension):
            paintings_dict[filename[:-4]] = paintings_dir + filename

    print(photos_dict)
    print(paintings_dict)


'''
Functions to visualise the input
'''


# Converts Tensor Object to Image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# Loads the Image
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# Displays the Image
def show_image(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


'''
Runs The ML Program Start To Finish
'''


def training(content, style):
    print("From NST", '{}-{}.jpg'.format(content, style))
    print("Training...")

    # Configuring Modules
    print("Creating TF image objects...")
    content_path = photos_dict[content]
    style_path = paintings_dict[style]
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Define Content and Style Representations
    print("Defining photo and painting input representations for the model...")
    x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    extractor = StyleContentModel(style_layers, content_layers)

    # Run Gradient Descent
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)      # This is the image object that gets passed to numerous functions

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_weight = 1e-2
    content_weight = 1e4
    total_variation_weight = 30

    input_values = [style_targets, style_weight, num_style_layers,
                    content_targets, content_weight, num_content_layers,
                    extractor, opt]

    '''
    Perform a Long Optimisation
    '''
    print("Performing Long Optimisation...")
    start = time.time()
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, input_values, total_variation_weight)
            print(".", end='')
        display.clear_output(wait=True)
        # display.display(tensor_to_image(image))
        print("Train step: {}".format(step))
    end = time.time()
    print("Total time: {:.1f}".format(end - start))
    print("Finished Optimisation!")

    # Total Variational Loss
    tf.image.total_variation(image).numpy()

    '''
    Rerun Optimisation
    '''
    print("Rerunning Optimisation With New Variational Loss Calculated...")
    image = tf.Variable(content_image)     # This is the redefined image object
    start = time.time()
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, input_values, total_variation_weight)
            print(".", end='')
        display.clear_output(wait=True)
        # display.display(tensor_to_image(image))
        print("Train step: {}".format(step))
    end = time.time()
    print("Total time: {:.1f}".format(end - start))
    print("Finished rerunning optimisation!")
    save_results(image, content, style)
    gen_path = generated_dir + '/detail/' + '{}-{}.jpg'.format(content, style)
    return gen_path, image


'''
Extracting Model Features Functions
'''


def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


# Calculate Style and Extracting Style and Content
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/num_locations


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, input_values):
    style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers, extractor, opt = input_values
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


'''
Training Function
'''


def train_step(image, input_values, total_variation_weight):
    extractor, opt = input_values[-2], input_values[-1]
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, input_values)
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


'''
Total Variation Loss Functions
'''


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


'''
Saving Results Function
'''


def save_results(image, content, style):
    print("Saving Results...")
    file_name = '{}-{}.jpg'.format(content, style)
    print("Saving {}...".format(file_name))
    tensor_to_image(image).save(generated_dir + file_name)
    print("Finished Saving Stylised File")
    print("Finished ML Process In ", datetime.now() - start_process)


# Builds a model that returns the style and content tensors
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # Expects float input in [0,1]
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


'''
Simple Command To Run The Whole Thing
'''


def run(content, style):
    initialize()
    gen_path, image = training(content, style)
    return gen_path, image


def run_all():
    initialize()
    for style in paintings_dict.keys():
        for content in photos_dict.keys():
            if not check_exists(content, style):
                training(content, style)
            else:
                print(content + '-' + style + '.jpg' + ' already has been generated!')


def check_exists(content, style):
    filename = content + '-' + style + '.jpg'
    for generated in os.listdir(generated_dir):
        if filename == generated:
            return True        # Has already been generated
    return False    #  Has not already been generated


