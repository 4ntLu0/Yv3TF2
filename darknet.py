# so we build the net here ? ? ?? ???? ?

'''
structure
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut] - skip connection
from=-3 | output of shortcut layer is obtained by adding feature maps from the previous to the third layer back (from the shortcut layer)
activation=linear

[upsample]
stride=2

ROUTE WTF IS ROUTE
[route]
layers = -4

[route]
layers = -1, 61
'''

