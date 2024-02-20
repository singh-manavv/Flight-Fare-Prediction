from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    Returns a list of requirements
    '''
    
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace('\n','') for requirement in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
    

setup(
    name='Flight-Fare-Prediction',
    version='0.0.1',
    author='singh-manavv',
    author_email='manvendras2608@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)