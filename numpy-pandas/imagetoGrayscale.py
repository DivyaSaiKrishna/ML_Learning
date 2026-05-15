import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#open image 
pandaImage = Image.open("public/panda.png")

#image to array
pandaImageArray = np.array(pandaImage)
print(pandaImageArray)

#Flip image
pandaImageFlip = pandaImageArray[::-1, : ,:]


#grayscale gray = 0.2989×R + 0.5870×G + 0.1140×B
r = pandaImageArray[ :, : ,0]
g = pandaImageArray[:, : , 1]
b = pandaImageArray[:, : , 2]

pandaGray = 0.2989 * r + 0.5870 * g + 0.1140 * b

#keep pixel values together
gray = np.clip(pandaGray, 0, 255).astype(np.uint8)

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(pandaImageArray)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pandaImageFlip)
plt.title("Flipped")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.tight_layout()
plt.show()


