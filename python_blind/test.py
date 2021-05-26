import sys

import numpy as np

from webrtc_aec3 import AEC3

def main(cmd_args):
    dir_AEC3 = dir(AEC3)
    print(f"dir of AEC3: {dir_AEC3}")
    aec3 = AEC3()
    a = np.array([1], dtype=np.int16)
    b = np.array([2], dtype=np.int16)
    print("prepare data")
    aec3.process_chunk(a, b)
    del aec3

if __name__ == "__main__":
    main(sys.argv[1:])
