from distutils.core import setup

setup(name='hybrid',
      version='0.1',
      description='An Implementation of Hybrid Student Modelling',
      author='Joseph Rollinson',
      author_email='jtrollinson@gmail.com',
      py_modules=['hybrid'],
      install_requires=[
          'numpy',
          'scipy',
          'pymc'])

