from setuptools import setup, find_packages

setup(
    name='CustomGPUMemoryAllocator',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pycuda',
        'cupy',
        'numba',
    ],
    author='Your Name',
    description='A custom GPU memory allocator for optimized GPU workloads',
    long_description=open('README.md').read(),
)
