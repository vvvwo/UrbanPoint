import numpy as np
import voxel_height

xyz = np.random.random((10000,3)) * 100

height = voxel_height.compute(xyz.astype(np.float32),0.6)
print(height)
