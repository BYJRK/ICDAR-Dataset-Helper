from pathlib import Path
from ic19 import IC19
from utils import get_info, mark_images, mark_images_multithreading


# ICDAR 2019 MLT Training Set
# 00001 - 01000:  Arabic
# 01001 - 02000:  English
# 02001 - 03000:  French
# 03001 - 04000:  Chinese
# 04001 - 05000:  German
# 05001 - 06000:  Korean
# 06001 - 07000:  Japanese
# 07001 - 08000:  Italian
# 08001 - 09000:  Bangla
# 09001 - 10000:  Hindi


if __name__ == "__main__":
    root = Path('D:/Datasets/ICDAR2019MLT/')
    train_img_path = root / 'train_img/'
    train_gt_path = root / 'train_gt/'
    output1 = root / 'train_img_arabic/'
    output2 = root / 'train_img_bangla/'
    output3 = root / 'train_img_hindi/'
    dataset = IC19(train_img_path, train_gt_path)
    # mark_images(dataset, output_path, shuffle=False, amount=3000)
    # print(get_info(dataset))
    # mark_images_multithreading(dataset, output1, range(1, 1001))
    # mark_images_multithreading(dataset, output2, range(8001, 9001))
    # mark_images_multithreading(dataset, output3, range(9001, 10001))
