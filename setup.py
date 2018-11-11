import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='regularized_differentiation',
    version='0.0.2',
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
    install_requires=[
        'numpy',
        'scipy',
        'cython>=0.29',  # implicit dependency of pyfftw
        'pyfftw',
    ],  # running examples will require scikit-image, jupyter
)
