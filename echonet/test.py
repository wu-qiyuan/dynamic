# Import the os module
import os

# Get the current working directory
cwd = os.getcwd()
cwd1 = os.path.dirname(cwd)

# Print the current working directory
print("Current working directory: {0}".format(cwd))

import sys

sys.path.append(cwd1)
import echonet.datasets as datasets
import echonet.utils as utils

# import echonet.utils.segmentation as utils

if __name__ == '__main__':
    utils.segmentation.run(output="/media/qiyuan/My_Passport/EchoNet-output/", save_video=True, batch_size=8)
