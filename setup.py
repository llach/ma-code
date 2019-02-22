from setuptools import setup
from setuptools import find_packages

setup(name='ma-code',
      version='0.0.1',
      description='code for masters thesis',
      author='Luca Lach',
      author_email='llach@teachfak.uni-bielefeld.de',
      url='https://github.com/llach/ma-code',
      install_requires=['numpy>=1.15.3',
                        'forkan>=0.0.1',
                        'gym>=0.10.9',
                        'scikit-image>=0.14.2',
                        'matplotlib>=3.0.0',
                        'tqdm>=4.30.0',
                        'baselines'],
      packages=find_packages())
