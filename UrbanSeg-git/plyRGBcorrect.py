import numpy as np
from plyfile import PlyData, PlyElement

# 读取PLY文件
plydata = PlyData.read('data/Yingrenshi.ply')

# 获取顶点数据
vertices = plydata['vertex'].data

# 转换RGB值为uint8
if 'red' in vertices.dtype.names and 'green' in vertices.dtype.names and 'blue' in vertices.dtype.names:
    vertices['red'] = (vertices['red']).astype(np.uint8)
    vertices['green'] = (vertices['green']).astype(np.uint8)
    vertices['blue'] = (vertices['blue']).astype(np.uint8)

# 保存为新的PLY文件
new_plydata = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
new_plydata.write('data/Yingrenshinew.ply')