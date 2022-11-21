import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.cocoeval import COCOeval


class COCOeval_X(COCOeval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def plot_pr(self, save_path_pr):
        pr_array1 = self.eval['precision'][0, :, 0, 0, 2]
        pr_array2 = self.eval['precision'][2, :, 0, 0, 2]
        pr_array3 = self.eval['precision'][4, :, 0, 0, 2]
        x = np.array(0.0, 1.01, 0.01)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)

        plt.plot(x, pr_array1, 'b-', label='IoU=0.5')
        plt.plot(x, pr_array2, 'c-', label='IoU=0.6')
        plt.plot(x, pr_array3, 'y-', label='IoU=0.7')

        plt.legend(loc='lower left')
        # plt.show()