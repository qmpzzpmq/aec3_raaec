import argparse
import sys
import os

from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='generate webrtc aec3')
    parser.add_argument('--webrtc_src', type=str, required=True)
    parser.add_argument('--webrtc_compiled', type=str, required=True)
    parser.add_argument('--CC', type=str, default="/usr/bin/clang")
    parser.add_argument('--CXX', type=str, default="/usr/bin/clang++")
    parser.add_argument('--WEBRTC_APM_DEBUG_DUMP', type=int, default=0)
    return parser

def main(cmd_args):
    parser = get_parser()
    args, setup_args = parser.parse_known_args(cmd_args)
    os.environ['CC'] = args.CC
    os.environ['CXX'] = args.CXX
    sys.argv[1:] = setup_args

    include_path = [
        args.webrtc_src,
        os.path.join(args.webrtc_src, "third_party", "abseil-cpp"),
        os.path.join(args.webrtc_src, "third_party", "tcmalloc"),
        os.path.join(args.webrtc_src, "third_party", "blink"),
        "/usr/include/glib-2.0",
        "/usr/lib/glib-2.0/include",
    ]

    extensions = [
        Extension(
            "webrtc_aec3",
            [
                'webrtc_aec3.pyx',
                os.path.join(
                    args.webrtc_src,
                    "modules/audio_processing/aec3/echo_canceller3.cc",
                ),
            ],
            include_dirs=include_path,
            library_dirs=[
                '/usr/lib/x86_64-linux-gnu',
                os.path.join(args.webrtc_compiled, "obj"),
                '/usr/lib',
                np.get_include(),
            ],
            libraries=[
                'webrtc',
                'pthread',
                'gthread-2.0',
                'glib-2.0',
            ],
            extra_compile_args=[
                "-std=c++14",
                "-stdlib=libc++",
                f"-D WEBRTC_APM_DEBUG_DUMP={args.WEBRTC_APM_DEBUG_DUMP}",
                "-D WEBRTC_POSIX",
            ],
            extra_link_args=[
                "-stdlib=libc++",
                "-lc++abi",
            ],
            language="c++",
            # define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
            # extra_compile_args=["-std=c++14",  "-v"],
            # extra_link_args=["-stdlib=libc++", ],
        )
    ]


    setup(
        ext_modules=
            cythonize(
                extensions,
                compiler_directives={'language_level' : "3"},
            ),
    )

if __name__ == "__main__":
    main(sys.argv[1:])
