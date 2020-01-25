# lists of paths that we should be using?

from absl import flags, logging
from abs1.flags import FLAGS

flags.DEFINE_string('weights', './utilities/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')

flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')