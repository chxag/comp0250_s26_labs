from setuptools import find_packages
from setuptools import setup

setup(
    name='coursework_world_spawner_lib',
    version='0.1.0',
    packages=find_packages(
        include=('coursework_world_spawner_lib', 'coursework_world_spawner_lib.*')),
)
