import sys

from webrtc_aec3 import AEC3

def main(cmd_args):
    dir_AEC3 = dir(AEC3)
    print(f"dir of AEC3: {dir_AEC3}")
    aec3 = AEC3()
    del aec3

if __name__ == "__main__":
    main(sys.argv[1:])
