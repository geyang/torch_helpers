from setuptools import setup

setup(name='torch_helpers',
      description='A set of helper functions for pyTorch',
      long_description='Moleskin makes it easy to print in terminals',
      version='0.1.0',
      url='https://github.com/episodeyang/torch_helpers',
      author='Ge Yang',
      author_email='yangge1987@gmail.com',
      license=None,
      keywords=['pyTorch', 'torch', 'deep learning', 'debugging'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3'
      ],
      packages=['torch_helpers'],
      install_requires=['moleskin']
      )
