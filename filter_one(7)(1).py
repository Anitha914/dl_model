from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
# load the model
model = VGG16()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# summarize filter shapes
print(model.layers)
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)