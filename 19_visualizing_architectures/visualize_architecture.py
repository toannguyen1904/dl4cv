from pyimagesearch.nn.conv import LeNet
from tensorflow.keras.utils import plot_model

model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True)