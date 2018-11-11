import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='regularized_differentiation',
    version='0.0.4',
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
        'scipy>=1.0',
        'pyfftw',  # ATTM need to pip install cython for pyfftw to install
    ],  # running examples will require scikit-image, jupyter
)
