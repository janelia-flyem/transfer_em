"""Save model and metadata to disk.

    % python save_model.py <model name> <ckpt dir> <mean_x> <stddev_x> <mean_y> <stddev_y> <image size> <id3d=1 or 0>
"""

from transfer_em.utils import save_model
import sys

""" example
model_name = "trained_example"
ckpt_dir = "./checkpoints/train_hemi2hemi_1/ckpt-1"
meanstd_x = (0.19801877, 0.1824518)
meanstd_y = (0.06743993, 0.37753862)
size = 132
is3d = True
"""

model_name = sys.argv[1]
ckpt_dir = sys.argv[2]
meanstd_x = (float(sys.argv[3]), float(sys.argv[4]))
meanstd_y = (float(sys.argv[5]), float(sys.argv[6]))
size = int(sys.argv[7])
is3d = True if sys.argv[8] == "1" else False

save_model(model_name, ckpt_dir, meanstd_x, meanstd_y, size, is3d)
