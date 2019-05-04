import glob
import numpy as np
from matplotlib import pyplot as plt

check_array = np.array(
    [[116, 17, 36],
     [152, 43,150],
     [106,141, 34],
     [ 69, 69, 69],
     [  2,  1,  3],
     [127, 63,126],
     [222, 52,211],
     [  2,  1,140],
     [ 93,117,119],
     [180,228,182],
     [213,202, 43],
     [ 79,  2, 80],
     [188,151,155],
     [  9,  5, 91],
     [106, 75, 13],
     [215, 20, 53],
     [110,134, 62],
     [  8, 68, 98],
     [244,171,170],
     [171, 43, 74],
     [104, 96,155],
     [ 72,130,177],
     [242, 35,231],
     [147,149,149],
     [ 35, 25, 34],
     [155,247,151],
     [ 85, 68, 99],
     [ 71, 81, 43],
     [195, 64,182],
     [146,133, 92]]
        )
for filename in glob.glob('*.png'):
    img = plt.imread(filename) * 255
    img = img[:, 256:, :]
    img = img.reshape((-1, 3))
    for i in np.unique(img, axis=0):
        ok = False
        for j in check_array:
            if sum((i - j) ** 2) == 0:
                ok = True
                break
        if not ok: raise SystemError(0)
