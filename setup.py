'''
The setup.py file is used to configure the packaging and installation of the developed 
lending club loan ANN-based python project.
'''

from setuptools import setup, find_packages
from typing import List

requirement_lst: List[str] = []
def get_requirements() -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    """
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                # ignore empty lines and "-e ." which is used for editable installs
                if not requirement or requirement.startswith('-e .'):
                    continue
                requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found. Please ensure it exists.")
    return requirement_lst

setup(
    name='lending_club_loan_ann',
    version='0.1',
    author='Hossein Beidaghydizaji',
    author_email='beidaghydizaji@gmx.de',
    packages=find_packages(),
    install_requires=get_requirements()
)