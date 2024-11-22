from pathlib import Path
from setuptools import setup

description = ['Training and evaluation of the PureACL']

with open(str(Path(__file__).parent / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

with open(str(Path(__file__).parent / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

extra_dependencies = ['jupyter', 'scikit-learn', 'ffmpeg-python', 'kornia']

setup(
    name='PureACL',
    version='1.0',
    packages=['PureACL'],
    python_requires='>=3.6',
    install_requires=dependencies,
    extras_require={'extra': extra_dependencies},
    author='shan wang',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/*/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
