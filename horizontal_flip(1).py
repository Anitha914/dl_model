# from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator
# from matplotlib import pyplot
# # load the image
# img = load_img('bird.jpg')
# # convert to numpy array
# data = img_to_array(img)
# # expand dimension to one sample
# samples = expand_dims(data, 0)
# # create image data augmentation generator
# datagen = ImageDataGenerator(horizontal_flip=True)
# # prepare iterator
# it = datagen.flow(samples, batch_size=1)
# # generate samples and plot
# for i in range(9):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# generate batch of images
# 	batch = it.next()
# 	# convert to unsigned integers for viewing
# 	image = batch[0].astype('uint8')
# 	# plot raw pixel data
# 	pyplot.imshow(image)
# # show the figure
# pyplot.show()


from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from numpy import expand_dims
import matplotlib.pyplot as plt

# Load image
img = load_img('bird.jpg')

# Convert to array
img_array = img_to_array(img)

# Add batch dimension
img_array = expand_dims(img_array, 0)

# Horizontal flip augmentation
datagen = ImageDataGenerator(horizontal_flip=True)

# Create generator
generator = datagen.flow(img_array, batch_size=1)

# Show 9 flipped images
plt.figure(figsize=(8, 8))
for i in range(9):
    flipped_img = generator.next()[0].astype('uint8')
    plt.subplot(3, 3, i+1)
    plt.imshow(flipped_img)
    plt.axis('off')

plt.show()
