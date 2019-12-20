from distutils.core import setup

setup(
    name='domain_randomizer',
    version='0.0.1',
    packages=['randomizer',],
    package_data={'randomizer': ['config/*/*']},
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    include_package_data=True
)
