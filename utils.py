from PIL import Image, ImageDraw
from pathlib import Path
from ic19 import *
from matplotlib import pyplot as plt
import progressbar
import numpy as np
from multiprocessing import pool


POINT_COLORS = ['red', 'yellow', 'lime', 'blue']
POLY_COLORS = ['teal', 'violet']
PALETTE = {
    'Latin': 'lightblue',
    'Chinese': 'lightblue',
    'Japanese': 'lightblue',
    'Korean': 'lightblue',
    'Symbols': 'lightblue',
    'None': 'lightblue',
    'Mixed': 'lightblue',
    'Arabic': 'red',
    'Hindi': 'blue',
    'Bangla': 'green'
}


def imshow(im):
    plt.figure()
    plt.imshow(im)
    plt.show()


def get_scale(im, minlength):
    w, h = im.size
    scale = minlength / min(w, h)
    return scale


def draw_points(canvas, labels):
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    # 根据图片大小，设定笔触的尺寸
    l = max(w, h) // 600
    l = max(l, 1)

    def point_to_circle(x, y):
        return [x - l, y - l, x + l, y + l]

    for label in labels:
        for point, color in zip(label.coor_pairs_tuple, POINT_COLORS):
            xy = point_to_circle(*point)
            draw.ellipse(xy, fill=color)

    return canvas


def draw_polys(canvas, labels):
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    # 框的线条粗细
    l = max(w, h) // 400
    l = max(l, 1)

    def draw_poly(draw, label, color):
        p1, p2, p3, p4 = label.coor_pairs_tuple
        draw.line([p1, p2, p3, p4, p1], width=l, fill=color)

    for label in labels:
        color = POLY_COLORS[1] if label.ignore else POLY_COLORS[0]
        draw_poly(draw, label, color)

    return canvas


def draw_polys_by_palette(canvas, labels):
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    # 框的线条粗细
    l = max(w, h) // 400
    l = max(l, 1)

    def draw_poly(draw, label, color):
        p1, p2, p3, p4 = label.coor_pairs_tuple
        draw.line([p1, p2, p3, p4, p1], width=l, fill=color)

    for label in labels:
        color = PALETTE[label.lang]
        draw_poly(draw, label, color)

    return canvas


def draw_anns(dataset, output_path, index):
    im = dataset.get_labels(index)
    labels = dataset.get_labels(index)
    draw_polys()


def mark_images_multithreading(dataset, output_path, indices):
    output_path.mkdir(exist_ok=True)
    for index in progressbar.progressbar(indices):
        im = dataset.get_img(index)
        labels = dataset.get_labels(index)
        draw_polys_by_palette(im, labels)
        file = output_path / Path(f'img_{str(index).zfill(5)}.jpg')
        im.save(file)


def mark_images(dataset, output_path, amount=5000, shuffle=True):
    if not output_path.exists() or not output_path.is_dir():
        output_path.mkdir()
    if shuffle:
        indices = np.array(range(dataset.range[0], dataset.range[1] + 1))
        np.random.shuffle(indices)
        indices = indices[:amount]
    else:
        indices = np.round(np.linspace(
            dataset.range[0], dataset.range[1], amount)).astype(np.int32)
    for index in progressbar.progressbar(indices):
        im = dataset.get_img(index)
        labels = dataset.get_labels(index)
        im = draw_polys_by_palette(im, labels)
        im = draw_points(im, labels)
        file = output_path / Path(f'img_{str(index).zfill(5)}.jpg')
        im.save(file)


def distance(p, q):
    return math.sqrt(math.pow(p[0] - q[0], 2) + math.pow(p[1] - q[1], 2))


def get_info(dataset):
    image_sizes = []
    image_aspect_ratios = []
    box_sizes = []
    box_relative_sizes = []
    box_aspect_ratios = []
    box_texts = []
    index_list = []
    try:
        for index in progressbar.progressbar(range(dataset.range[0], dataset.range[1] + 1)):
            im = dataset.get_img(index)
            labels = dataset.get_labels(index)
            w, h = im.size
            image_sizes.append([w, h])
            image_aspect_ratios.append(w / h)
            length = max(w, h)
            for label in labels:
                box_sizes.append([label.width, label.height])
                box_relative_sizes.append(
                    [label.width / length, label.height / length])
                box_aspect_ratios.append([label.aspect_ratio])
                if not label.aspect_ratio:
                    print(
                        f'in pic {index}, text box {label.text} has zero aspect ratio')
                box_texts.append(label.text)
                # 用来标注第 n 个文本框在第 i 张图片中
                index_list.append(index)
    except Exception as e:
        print(index, e)

    np.save('image_sizes', np.array(image_sizes))
    np.save('image_aspect_ratios', np.array(image_aspect_ratios))
    np.save('box_sizes', np.array(box_sizes))
    np.save('box_relative_sizes', np.array(box_relative_sizes))
    np.save('box_aspect_ratios', np.array(box_aspect_ratios))
    np.save('index_list', np.array(index_list))
    print('data saved.')
