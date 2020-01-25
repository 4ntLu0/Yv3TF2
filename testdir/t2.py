from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import t1


def main(_argsv):
    print(FLAGS.weights)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
