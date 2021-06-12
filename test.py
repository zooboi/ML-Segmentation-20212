# Get RGB
color = '#bdbdbd'
h = color.lstrip('#')
rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
print(rgb[0])
