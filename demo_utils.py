from __future__ import print_function
from ipywidgets import fixed
import ipywidgets as widgets
from IPython.display import display, Image


def display_demo(use_cached=True):
    if not use_cached:
        ## using global imports only for uncached data
        global plt, mpimg, np, tf, Activation, Conv2D, Dense
        global Flatten, Input, MaxPooling2D, SparseCategoricalCrossentropy
        global load_model, Model, InternalInfluence, get_model_wrapper
        global ChannelMaskVisualizer, HeatmapVisualizer, Tiler

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np
        import tensorflow as tf

        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Input
        from tensorflow.keras.layers import MaxPooling2D
        from tensorflow.keras.losses import SparseCategoricalCrossentropy
        from tensorflow.keras.models import load_model
        from tensorflow.keras.models import Model

        from trulens.nn.attribution import InternalInfluence
        from trulens.nn.models import get_model_wrapper
        from trulens.visualizations import ChannelMaskVisualizer
        from trulens.visualizations import HeatmapVisualizer
        from trulens.visualizations import Tiler

        tf.keras.backend.set_image_data_format('channels_last')

        # Allow memory growth to avoid tensorflow taking all the RAM.
        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
        setup_data_and_model()

    model_widget = widgets.Select(
        options=['Pink', 'No_Pink'],
        value='Pink',
        # rows=10,
        description='Model:',
        disabled=False)
    instance_widget = widgets.Select(
        options=[
            'Tony_Blair', 'Gerhard_Schroeder', 'Gerhard_Schroeder_Edited'
        ],
        value='Tony_Blair',
        # rows=10,
        description='Instance:',
        disabled=False)
    visualizer_widget = widgets.Select(
        options=['heatmap', 'channel'],
        value='heatmap',
        # rows=10,
        description='Visualizer:',
        disabled=False)
    widget_group = widgets.HBox(
        [model_widget, instance_widget, visualizer_widget])
    out = widgets.interactive_output(
        visualize_lfw, {
            'keras_model_name': model_widget,
            'instance_name': instance_widget,
            'use_cached': fixed(use_cached),
            'layer': fixed(4),
            'visualizer_name': visualizer_widget
        })
    display(widget_group, out)


def setup_data_and_model():
    global all_data, x_train, y_train, x_test, y_test, data_name_dict
    global keras_model, keras_model_no_pink, keras_model_name_dict
    global model_wrapper, model_wrapper_no_pink, model_wrapper_name_dict
    global keras_model, keras_model_no_pink
    all_data, x_train, y_train, x_test, y_test = load_lfw_data()

    data_name_dict = {
        'Tony_Blair': all_data['pink_in_tr'],
        'Gerhard_Schroeder': all_data['gerhard'],
        'Gerhard_Schroeder_Edited': all_data['gerhard_edited']
    }
    keras_model = load_model('resources/model_with_pink.h5')
    keras_model_no_pink = load_model('resources/model_no_pink.h5')
    keras_model_name_dict = {
        'Pink': keras_model,
        'No_Pink': keras_model_no_pink
    }

    model_wrapper = get_model_wrapper(keras_model)
    model_wrapper_no_pink = get_model_wrapper(keras_model_no_pink)
    model_wrapper_name_dict = {
        'Pink': model_wrapper,
        'No_Pink': model_wrapper_no_pink
    }

    keras_model = load_model('resources/model_with_pink.h5')
    keras_model_no_pink = load_model('resources/model_no_pink.h5')


def load_lfw_data(include_pink=True):
    # Load our data.
    all_data = np.load('resources/lfw.npz')
    if include_pink:
        x_train = all_data['x_tr_pink']
        y_train = all_data['y_tr_pink']
    else:
        x_train = all_data['x_tr_no_pink']
        y_train = all_data['y_tr_no_pink']
    x_test = all_data['x_te']
    y_test = all_data['y_te']
    return all_data, x_train, y_train, x_test, y_test


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def plot_prediction(keras_model, instance, num_classes=5):
    predictions = keras_model.predict(instance)
    return predictions.argmax(-1)


def show_image(image_path):
    display(Image(filename=image_path))


def visualize_lfw(keras_model_name,
                  instance_name,
                  use_cached,
                  layer=4,
                  visualizer_name='heatmap'):
    if use_cached:
        show_image('cached_images/{}_{}_original.jpg'.format(
            keras_model_name, instance_name))
        show_image('cached_images/{}_{}_{}.jpg'.format(keras_model_name,
                                                       instance_name,
                                                       visualizer_name))
    else:
        keras_model = keras_model_name_dict[keras_model_name]
        model = model_wrapper_name_dict[keras_model_name]
        instance = data_name_dict[instance_name]
        predicted_class = plot_prediction(keras_model, instance)
        plt.axis('off')
        plt.imshow(Tiler().tile(instance))
        plt.title('Original Image, Predicted class: {}'.format(' '.join(
            map(str, predicted_class))))
        plt.savefig('cached_images/{}_{}_original.jpg'.format(
            keras_model_name, instance_name))
        layer = 4

        # Define the influence measure.
        internal_infl_attributer = InternalInfluence(model,
                                                     layer,
                                                     qoi='max',
                                                     doi='point')

        internal_attributions = internal_infl_attributer.attributions(instance)

        # Take the max over the width and height to get an attribution for each channel.
        channel_attributions = internal_attributions.max(axis=(1,
                                                               2)).mean(axis=0)

        target_channel = int(channel_attributions.argmax())
        if visualizer_name == 'heatmap':
            # Calculate the input pixels that are most influential on the target channel.
            fig = plt.figure()
            input_attributions = InternalInfluence(
                model, (0, layer), qoi=target_channel,
                doi='point').attributions(instance)

            # Visualize the influential input pixels.
            _ = HeatmapVisualizer(blur=3)(input_attributions,
                                          instance,
                                          fig=fig,
                                          imshow=False)
            plt.title('Explanation')

        elif visualizer_name == 'channel':
            plt.figure()

            visualizer = ChannelMaskVisualizer(model,
                                               layer,
                                               target_channel,
                                               blur=3,
                                               threshold=0.9)

            visualization = visualizer(instance)
            plt.axis('off')
            plt.imshow(Tiler().tile(visualization))
            plt.title('Explanation')
        plt.savefig('cached_images/{}_{}_{}.jpg'.format(
            keras_model_name, instance_name, visualizer_name))
        plt.show()
