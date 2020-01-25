from absl import logging
import numpy as np
import tensorflow as tf
import cv2

YOLO_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

def test_function():
    print('test_function called')

def load_weights(model, weights_path):
    weights = open(weights_path, 'rb')
    major, minor, revision, seen, _ = np.fromfile(weights, dtype=np.int32, counts=5)

    layers = YOLO_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layers(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(sub_model.name, layer_name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            input_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(weights, dtype=np.float32, count=filters)
            else:
                # batch norm weights
                bn_weights = np.fromfile(weights, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out, in, height, width)
            conv_shape = (filters, input_dim, size, size)
            conv_weights = np.fromfile(weights, dtype=np.float32, count=np.product(conv_shape))
            #tf shape (height, width, input, output)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)
    assert len(weights.read()) == 0, 'failed to read all data'
    weights.close()