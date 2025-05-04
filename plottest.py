from PIL import Image, ImageDraw

# Load the image
image_path = r"C:\Users\Mike\Desktop\Untitled.png"
image = Image.open(image_path)

# Coordinates to draw (left, top, right, bottom)
box_coords = (1011.0, 51.0, 1080.0, 100.5) 

# Draw a rectangle
draw = ImageDraw.Draw(image)
draw.rectangle(box_coords, outline="red", width=3)

# Show the result
image.show()