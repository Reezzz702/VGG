from skimage import io
import numpy as np
img = io.imread('dataset/train/10801.jpg')
print(type(img))

a = img.reshape(3,224,224)
b = a.reshape(224,224,3)

print(np.equal(img,b))