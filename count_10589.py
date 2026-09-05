import cv2, numpy as np

l1 = cv2.imread('C:/satquery/label1/10589.png')
l2 = cv2.imread('C:/satquery/label2/10589.png')

# Count colors in l1 and l2
def color_counts(img):
    pixels = img.reshape(-1, 3)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    for u, cnt in zip(unique, counts):
        print(f"RGB: ({u[2]}, {u[1]}, {u[0]}) -> Count: {cnt} ({cnt/len(pixels)*100:.2f}%)")

print("label1:")
color_counts(l1)
print("label2:")
color_counts(l2)
