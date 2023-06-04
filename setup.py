from setuptools import setup, find_packages

setup(
    name='datamodules',
    version='0.0.1',
    description='A package to share some datamodules in for pythorch lightning',
    author='Guido Gagliardi',
    author_email='guido.gagliardi@phd.unipi.it',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn-intelex',
        'lightning',
        'torch',
        'torchvision',
        'torchaudio'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
