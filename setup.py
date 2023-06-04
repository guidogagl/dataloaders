from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1.0',
    description='Descrizione del tuo pacchetto',
    author='Il tuo nome',
    author_email='tua@email.com',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'torch',
        'pytorch-lightning'
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
