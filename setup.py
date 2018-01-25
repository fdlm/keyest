from setuptools import setup

setup(
    name='keyest',
    version='0.0.1',
    packages=['keyest'],
    url='',
    license='MIT',
    author='Filip Korzeniowski',
    author_email='filip.korzeniowski@jku.at',
    description='',
    requires=['numpy', 'madmom', 'trattoria', 'Lasagne', 'Theano', 'tqdm',
              'termcolor', 'docopt']
)
