#%%
import numpy as np
from matplotlib import pyplot as plt

#%%
# 图片尺寸统计
image_sizes = np.load('ic19/image_sizes.npy')
print('Average image size:', np.mean(image_sizes, axis=0))
plt.figure()
plt.hist(np.max(image_sizes, axis=1), bins=30, rwidth=0.6)
plt.xlim(0, 6000)
plt.title('Image size distribution')
plt.savefig('image size.png')
plt.show()

#%%
# 图片宽高比统计
image_ar = np.load('ic19/image_aspect_ratios.npy')
print('Average image aspect ratio:', np.mean(image_ar))
plt.figure()
plt.hist(image_ar, bins=30, rwidth=0.6, range=(0,5))
plt.xlim(0, 5)
plt.xticks(np.linspace(0,5,11))
plt.title('Image aspect ratio distribution')
plt.xlabel('aspect ratio (width / height)')
plt.ylabel('amount')
plt.savefig('image aspect ratio.png')
plt.show()

#%%
# 文本框尺寸统计
box_sizes = np.load('ic19/box_sizes.npy')
print('Average box size:', np.mean(box_sizes, axis=0))
# print('Min box size:', np.min(box_sizes[box_sizes>0]))
print('Max box size:', np.max(box_sizes))
plt.figure()
plt.hist(np.max(box_sizes, axis=1), bins=15, rwidth=0.6, range=(0, 1500))
plt.xticks(np.linspace(0, 1500, 16))
plt.xlim(0, 1500)
plt.title('Box size distribution')
plt.grid(True)
plt.xlabel('box width')
plt.ylabel('amount')
plt.savefig('box size.png')
plt.show()

#%%
box_relative_sizes = np.load('ic19/box_relative_sizes.npy')
print('Average box relative size:', np.mean(box_relative_sizes, axis=0))
print('Max box relative size:', np.max(box_relative_sizes))
plt.figure()
plt.hist(np.max(box_relative_sizes, axis=1), bins=20, rwidth=0.6)
plt.xticks(np.linspace(0, 1, 11))
plt.xlim(0, 1)
plt.xlabel('box relative size')
plt.ylabel('amount')
plt.title('Box relative sizes distribution')
plt.savefig('box relative size.png')
plt.show()

#%%
box_ar = np.load('ic19/box_aspect_ratios.npy')
# box_sizes = np.load('box_sizes.npy')
# eps = 1e-5
# box_ar = np.max(box_sizes, axis=1) / (np.min(box_sizes, axis=1) + eps)
print('Average box aspect ratio:', np.mean(box_ar))
plt.figure()
plt.hist(box_ar, bins=25, range=(0, 20), rwidth=0.6)
plt.xlim(0, 20)
plt.title('Box aspect ratio distribution')
plt.xlabel('box aspect ratio')
plt.ylabel('amount')
plt.grid(True)
plt.savefig('box aspect ratio.png')
plt.show()
