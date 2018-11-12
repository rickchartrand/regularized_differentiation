import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='regularized_differentiation',
    version='0.0.5',
    author='Rick Chartrand',
    author_email='rick@descarteslabs.com',
    description='regularized numerical differentiation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rickchartrand/regularized_differentiation',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    setup_requires=[
        'numpy>=1.6',
        'scipy>=1.0',
        'cython>=0.28',
    ],
    install_requires=[
        'numpy>=1.6',
        'scipy>=1.0',
        'cython',
        'pyfftw',  # ATTM need to pip install cython for pyfftw to install
    ],  # running examples will require scikit-image, jupyter
)
