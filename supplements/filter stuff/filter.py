from PIL import Image, ImageFilter

# Open image
image = Image.open("tomorrowland.jpeg").convert("RGB")
image.show()

# Filter image according to edge detection kernel
filtered = image.filter(ImageFilter.Kernel(
    size=(3, 3),
    kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
    scale=1
))

# Show resulting image
filtered.show()