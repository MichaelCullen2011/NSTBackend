import tensorflow as tf

import matplotlib as mpl
import numpy as np
import os
import PIL.Image
import csv
from datetime import datetime

start = datetime.now()
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False


'''
Defining Model and Image Paths
'''
print("Finding images and model roots...")
# root_dir = 'C:/Users/micha/Documents/PycharmProjects/NST_Lite/'
root_dir = os.getcwd()

# Defining Model DIR
style_predict_path = root_dir + '\\models\\inceptionv3_fp16_predict.tflite'
style_transform_path = root_dir + '\\models\\inceptionv3_fp16_transfer.tflite'

# Iterates through all content and style images and adding to dict
content_dir = root_dir + '\\images\\content\\'
style_dir = root_dir + '\\images\\styles\\'
generated_dir = root_dir + '\\images\\generated\\'

content_paths = {}
style_paths = {}

for content_name in os.listdir(content_dir):
    if content_name.endswith(".jpg") or content_name.endswith(".png"):
        content_paths[content_name[:-4]] = content_dir + content_name
    else:
        continue

for content_name in os.listdir(style_dir):
    if content_name.endswith(".jpg") or content_name.endswith(".png"):
        style_paths[content_name[:-4]] = style_dir + content_name
    else:
        continue

print(style_paths)


def read_csv(file):
    print("Reading content and style images to use in model...")
    txt = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for name in row:
                txt.append(name.strip())
    content, style = txt[0], txt[1]
    return content, style


'''
Preprocess Inputs
'''


# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
    print("Loading Image...")
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


# Function to pre-process by resizing and central cropping it.
def preprocess_image(image, target_dim):
    print("Preprocessing Image...")
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    return image


'''
Run Style Transfer      -       Run Style Predict Model
                                Run Style Transform Model
'''


# Runs style prediction on the preprocessed style image
def run_style_predict(preprocessed_style_image):
    print("Running style bottleneck...")
    # Load Model
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)

    # Set input
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], preprocessed_style_image)

    # Calculate bottleneck
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]['index']
    )()

    return style_bottleneck


# Run style transform on the preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image):
    print("Running transformation...")
    # Load Model
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)

    # Set input
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]['index']
    )()
    
    return stylized_image


'''
Style Blending      -       This will now use the stylized image as the content
                            image and the original content image as the image
                            to style from; resulting in blending backwards
'''


def style_blending(content_image, style_bottleneck, preprocessed_content_image):
    print("Blending transformation...")
    # Calculate style bottleneck of the content image.
    style_bottleneck_content = run_style_predict(
        preprocess_image(content_image, 256)
    )

    # Define blending ratio
    content_blending_ratio = 0.3

    # Blend the image
    style_bottleneck_blended = content_blending_ratio * style_bottleneck_content\
                               + (1 - content_blending_ratio) * style_bottleneck

    # Stylize the content image using the style bottleneck.
    stylized_image_blended = run_style_transform(style_bottleneck_blended,
                                                 preprocessed_content_image)

    return stylized_image_blended


'''
Saving Our Images
'''


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def save_results(final_image, content_to_use, style_to_use, blended=False):
    print("Saving Results...")
    if blended:
        file_name = '{}-{}.jpg'.format(content_to_use, style_to_use)
        tensor_to_image(final_image).save(generated_dir + '\\blended\\' + file_name)
        tensor_to_image(final_image).close()
    else:
        file_name = '{}-{}.jpg'.format(content_to_use, style_to_use)
        tensor_to_image(final_image).save(generated_dir + '\\lite\\' + file_name)
        tensor_to_image(final_image).close()


def generate(content, style, csv=False, blended=False):
    # Chooses an image from our content and style dict
    if csv:
        content_to_use, style_to_use = read_csv('images_to_use.txt')
    else:
        content_to_use, style_to_use = content, style

    content_to_use = content
    style_to_use = style
    print("Transferring Style: ", style_to_use, "to content: ", content_to_use)
    content_path = content_paths[content_to_use]
    style_path = style_paths[style_to_use]

    # Runs the Load function and saves the input images.
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Runs the Preprocess function and saves the input images.
    preprocessed_content_image = preprocess_image(content_image, 384)
    preprocessed_style_image = preprocess_image(style_image, 256)

    # Runs the function and saves the bottleneck
    style_bottleneck = run_style_predict(preprocessed_style_image)

    if blended:
        stylized_image_blended = style_blending(content_image, style_bottleneck, preprocessed_content_image)
        save_results(stylized_image_blended, content_to_use, style_to_use, blended=True)

    else:
        # Runs the transform function and saves the image
        stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)
        save_results(stylized_image, content_to_use, style_to_use)

    stylized_image = tensor_to_image(stylized_image)
    saved_path = generated_dir + '\\lite\\' + '{}-{}.jpg'.format(content_to_use, style_to_use)
    return saved_path, stylized_image


# Iterates through all style and content images, generating transferred styles for all
def nst_all():
    for style in style_paths.keys():
        for content in content_paths.keys():
            generate(content, style, csv=False, blended=True)


# Just use style and content in the csv file
def nst_csv(content, style):
    gen_path, image = generate(content=content, style=style, csv=True, blended=False)
    print("Finished Process In ", datetime.now() - start)
    return gen_path, image



