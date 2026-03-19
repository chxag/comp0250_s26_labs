from setuptools import find_packages
from setuptools import setup

setup(
    name='cw1_team_x',
    version='0.1.0',
    packages=find_packages(
        include=('cw1_team_x', 'cw1_team_x.*')),
)
