from setuptools import setup, find_packages

# run this on terminal
# python setup.py sdist bdist_wheel

setup(
    name = 'swtipy',
    version = '0.1',
    packages = find_packages(),
    install_requires = [
        'numpy >= 2.0',
        'numba >= 0.6.2'
    ],

)