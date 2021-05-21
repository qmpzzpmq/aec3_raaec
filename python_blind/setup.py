import argparse
import sys
import os

from setuptools import setup, Extension
from Cython.Build import cythonize

def get_parser():
    parser = argparse.ArgumentParser(description='generate webrtc aec3')
    parser.add_argument('--webrtc_src', type=str, required=True)
    parser.add_argument('--webrtc_compiled', type=str, required=True)
    return parser

def main(cmd_args):
    parser = get_parser()
    args, setup_args = parser.parse_known_args(cmd_args)
    sys.argv[1:] = setup_args

    extensions = [
        Extension(
            "webrtc_aec3",
            ["webrtc_aec3.pyx"], extra_compile_args=["-std=c++14"])
    ]

    setup(
        ext_modules=
            cythonize(extensions,
                compiler_directives={'language_level' : "3"},),
        include_dirs = [
            args.webrtc_src,
            os.path.join(args.webrtc_src, "third_party", "abseil-cpp"),
            os.path.join(args.webrtc_src, "third_party", "tcmalloc"),
            os.path.join(args.webrtc_src, "third_party", "blink"),
            "/usr/include/glib-2.0",
            "/usr/lib/glib-2.0/include",
            ],
    )

if __name__ == "__main__":
    main(sys.argv[1:])