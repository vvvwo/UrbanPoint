from utils.helper_ply import read_ply, write_ply
import glob
import os
import numpy as np

vis_path = "result\\urban_seg\\save\\"
output = "result\\urban_seg\\save\\"
colors_map = np.array(
    [[211, 220, 235], [77, 179, 63], [85, 195, 233], [243, 151, 29], [152, 31, 35], [42, 102, 177], [220, 188, 144]])
for path in glob.glob(os.path.join(vis_path, "*.ply")):
    name = os.path.basename(path)[:-4]
    data = read_ply(path)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    labels = data['pred'].astype(np.uint8)
    colors = colors_map[labels].astype(np.uint8)
    write_ply(output + name + "_rgb", [points, colors],
              ['x', 'y', 'z', 'red', 'green', 'blue'])
    if not os.path.exists(output + "_pred.txt"):
        np.savetxt(output + name + "_pred.txt", labels, fmt='%i', delimiter=' ')
