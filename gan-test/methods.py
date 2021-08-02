import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

def draw_img_gallery(data, rows = 0, width = 400, labels = None):
    lines = 0
    img_count = len(data)
    if rows <= 0 or rows % 1 > 0:
        rows = math.sqrt(img_count) // 1

    lines = img_count // rows
    if img_count % rows > 0:
        lines += 1
    width / 80, width * lines / (80 * rows)
    plt.figure(dpi=80, figsize = (width / 80, width * lines / (80 * rows)))

    plt.rcParams['font.sans-serif'] = ['YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    for img_index in range(img_count):
        plt.subplot(int(lines), int(rows), img_index + 1)
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.4, hspace=0.4)
        plt.axis('off')
        if labels:
            plt.title(labels[img_index])
        plt.imshow(data[img_index])
    plt.show()

import reportlab.graphics.shapes