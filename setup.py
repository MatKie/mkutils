from setuptools import setup, find_packages

setup(
    name='mkutils',
    version='0.2.0',
    url='https://tbd.com',
    author='Matthias Kiesel',
    author_email='m.kiesel18@imperial.ac.uk',
    description='Utility Modules for evaluating molecular simulation and odd jobs',
    packages=find_packages(),    
    install_requires=['numpy'],
)
