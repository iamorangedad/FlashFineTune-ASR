from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 针对 Jetson 的优化参数
cxx_args = ["-O3"]
nvcc_args = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_87,code=sm_87",  # Orin Nano 的架构是 sm_87
]

setup(
    name="fast_mel_cuda",  # 编译后的包名
    ext_modules=[
        CUDAExtension(
            name="fast_mel_cuda_impl",  # 模块内部名称
            sources=["audio/fast_mel.cu"],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
