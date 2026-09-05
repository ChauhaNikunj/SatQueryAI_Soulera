import cv2, glob, numpy as np
from collections import Counter

# Let's inspect 20 label pairs
l1_files = glob.glob('C:/satquery/label1/*.png')[:20]

# Check color values in label1 and label2
color_to_id = {}
all_colors = set()

for f in l1_files:
    fname = f.split('\\')[-1]
    im1 = cv2.imread(f'C:/satquery/label1/{fname}')
    im2 = cv2.imread(f'C:/satquery/label2/{fname}')
    for c in np.unique(im1.reshape(-1, 3), axis=0):
        all_colors.add(tuple(c))
    for c in np.unique(im2.reshape(-1, 3), axis=0):
        all_colors.add(tuple(c))

print("Total unique colors found:", len(all_colors))
for c in all_colors:
    print("BGR:", c, "RGB:", (c[2], c[1], c[0]))
