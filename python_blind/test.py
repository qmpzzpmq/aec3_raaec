import sys
import argparse
import time
from str2bool import str2bool

import numpy as np
import soundfile as sf

from webrtc_aec3 import AEC3

def get_parser():
    parser = argparse.ArgumentParser(description='test case')
    parser.add_argument("ref", type=str)
    parser.add_argument("rec", type=str)
    parser.add_argument("linear", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--pure_linear", type=str2bool, default="false")
    parser.add_argument("--dtype", default='float', choices=['int', 'float'])
    return parser

def main(cmd_args): 
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    print(f"reading ref from {args.ref}")
    ref, fs = sf.read(args.ref, dtype="int16")
    assert fs == args.fs, \
        f"the ref {args.ref} fs {fs} is not equal the working fs {args.fs}"
    print(f"reading rec from {args.rec}")
    rec, fs = sf.read(args.rec, dtype="int16")
    assert fs == args.fs, \
        f"the rec {args.ref} fs {fs} is not equal the working fs {args.fs}"

    print("building AEC3 obj")
    aec3 = AEC3(fs=args.fs, pure_linear=args.pure_linear)
    print("AEC3 runing")
    time_start=time.time()

    if args.dtype == 'int':
        linear, out = aec3.run_int16(
            ref, rec,
        )
    else:
        ref = np.array(ref.astype(dtype=np.float32), order='C')
        rec = np.array(rec.astype(dtype=np.float32), order='C')
        linear, out = aec3.run_float(
            ref, rec,
        )
        # linear = np.array(linear * 32768, dtype=np.int16)
        # out = np.array(out * 32768, dtype=np.int16)
    time_end=time.time()
    print(
        f'''totally cost {time_end-time_start} with 
        length {min(len(ref), len(rec))} with pure linear {args.pure_linear}''')

    print(f"writing linear from {args.linear}")
    sf.write(args.linear, linear.squeeze(), args.fs)
    print(f"writing out from {args.out}")
    sf.write(args.out, out.squeeze(), args.fs)

if __name__ == "__main__":
    main(sys.argv[1:])
