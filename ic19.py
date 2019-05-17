from pathlib import Path
from PIL import Image
import math
import numpy as np
from functools import reduce
from itertools import chain
from shapely.geometry import Polygon


def distance(p1, p2):
    """
    计算两个点的欧几里得距离
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def flatten(mat):
    """
    将二维数组扁平化为一维数组
    """
    # return np.array(mat).flatten().tolist()
    return list(chain.from_iterable(mat))


class TextBox:
    """文本框"""

    def __init__(self, line):
        """
        传入 ICDAR 2015/2017MLT/2019MLT 标签文本的某一行

        形如 495,537,705,506,701,539,487,576,Latin,Division
        """
        # line
        splits = line.split(',')
        self.coors = list(map(int, splits[:8]))
        self.lang = splits[8]
        self.text = splits[-1]

    def scale_coors(self, scale):
        """
        将文本框坐标按照图片缩放的比率进行缩放
        """
        self.coors = [round(x * scale) for x in self.coors]

    def sort_poly(self):
        """
        排列四边形的四个顶点，保证顺序是从左上角开始按顺时针旋转
        """
        quad = np.array(self.coor_pairs_list)
        if self.area < 0:
            quad = quad[[0, 3, 2, 1]]
        # 排列组合四种情况
        x1, y1 = quad[0]
        x2, y2 = quad[1]
        x3, y3 = quad[2]
        x4, y4 = quad[3]
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)
        rect_coors = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        combination = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                       [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                       [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                       [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
        current_max = math.inf
        result_index = -1
        for i in range(4):
            temp = reduce(lambda x, y: x + y,
                          [distance(x, y) for x, y in zip(combination[i], rect_coors)])
            print(i, temp)
            if temp < current_max:
                current_max = temp
                result_index = i
        self.coors = flatten(combination[result_index])

    @property
    def coor_pairs_list(self):
        """
        返回形如 [[x1, y1], [x2, y2], ... , [x4, y4]] 的坐标列表
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = self.coors
        return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    @property
    def coor_pairs_tuple(self):
        """
        返回形如 [(x1, y1), (x2, y2), ..., (x4, y4)] 的坐标列表
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = self.coors
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    @property
    def ignore(self):
        """是否为难样本"""
        return self.text == '###'

    @property
    def width(self):
        """
        文本框的平均宽度
        """
        p1, p2, p3, p4 = self.coor_pairs_tuple
        l1 = distance(p1, p2)
        l3 = distance(p3, p4)
        return (l1 + l3) / 2

    @property
    def height(self):
        """
        文本框的平均高度
        """
        p1, p2, p3, p4 = self.coor_pairs_tuple
        l2 = distance(p2, p3)
        l4 = distance(p4, p1)
        return (l2 + l4) / 2

    @property
    def aspect_ratio(self):
        """文本框的长宽比"""
        # 这里为了简便，找出最长和最短的两条边，分别求平均数，然后相除
        if self.height != 0:
            return self.width / self.height
        else:
            return 0

    @property
    def bbox(self):
        """
        返回文本框的外围矩形包络框 [x, y, w, h]
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = self.coors
        left = min(x1, x2, x3, x4)
        right = max(x1, x2, x3, x4)
        top = min(y1, y2, y3, y4)
        bottom = max(y1, y2, y3, y4)
        width = right - left
        height = bottom - top
        return [left, top, width, height]

    @property
    def rect(self):
        """
        返回文本框的外围矩形包络框 [x1, y1, x2, y2]
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = self.coors
        left = min(x1, x2, x3, x4)
        right = max(x1, x2, x3, x4)
        top = min(y1, y2, y3, y4)
        bottom = max(y1, y2, y3, y4)
        return [left, top, right, bottom]

    @property
    def bbox_area(self):
        """
        计算外部包络框的面积
        """
        *_, w, h = self.bbox
        return w * h

    @property
    def area(self):
        """
        计算四边形框的实际面积
            用 Shoelace formula 计算四边形面积
            https://en.wikipedia.org/wiki/Shoelace_formula
            如果顺时针，则输出为正数，否则为负数
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = self.coors
        return (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1
                - x2 * y1 - x3 * y2 - x4 * y3 - x1 * y4) / 2


class IC19:
    """ICDAR 2019 MLT 数据集"""

    def __init__(self, img_path, gt_path):
        self.img_path = img_path
        self.gt_path = gt_path
        assert self.img_path.is_dir() and self.gt_path.is_dir()
        self.range = (1, 10000)

    def get_img(self, index):
        """
        根据给定的序号，获取 Image 格式的图片（会将格式不为 .jpg 的转为 RGB）
        """
        assert self.range[0] <= index <= self.range[1]
        # 图片名称形如 tr_img_00010.jpg
        file = self.img_path / Path(f'tr_img_{str(index).zfill(5)}.jpg')
        # print(file.absolute())
        if not file.exists():
            file = file.with_suffix('.png')
            if not file.exists():
                file = file.with_suffix('.gif')
                if not file.exists():
                    raise IndexError()
        im = Image.open(file)
        im = im.convert('RGB')
        return im

    def get_labels(self, index):
        assert self.range[0] <= index <= self.range[1]
        # 标签名称形如 tr_img_00010.txt
        file = self.gt_path / Path(f'tr_img_{str(index).zfill(5)}.txt')
        # print(file.absolute())
        if not file.exists():
            raise IndexError()
        # encoding='utf-8-sig'
        labels = []
        with file.open('r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.rstrip()
                textbox = TextBox(line)
                labels.append(textbox)
        return labels


if __name__ == "__main__":
    pass
