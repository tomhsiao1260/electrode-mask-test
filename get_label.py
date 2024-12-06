import argparse
import tifffile
import numpy as np
from utils.draw import drawImage

# inklabels generation
def getInklabel(sdf_path, label_path, output_path):
    # z, y, x
    dStack = tifffile.imread(sdf_path)
    labelStack = tifffile.imread(label_path)

    # normalized
    dStack = dStack.astype(np.float64) / 65535
    labelStack = labelStack.astype(np.float64) / 65535

    # you can change the weights here to fit your use case
    weight = np.maximum((1 - 1000.0 * dStack ** 2), 0)
    inklabels = weight * labelStack

    inklabels = (65535 * inklabels).astype(np.uint16)
    tifffile.imwrite(output_path, inklabels)
    drawImage(output_path)

# python get_label.py --sdf output/sdf.tif --label output/label.tif --o output/inklabels.tif
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate volume via boudning box')
    parser.add_argument('--sdf', type=str, help='SDF input path')
    parser.add_argument('--label', type=str, help='Labels input path')
    parser.add_argument('--o', type=str, help='Inklabels output path')
    args = parser.parse_args()

    getInklabel(args.sdf, args.label, args.o)