from setuptools import find_packages,setup
from typing import  List

hyphen_e="-e ."
def get_requirements(filepath: str)-> List[str]:
    requirements=[]

    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements=[ i.replace("\n","") for i in requirements]

        if hyphen_e in requirements:
            requirements.remove(hyphen_e)

setup(
    name="Loan_Prediction_Project",
    version="0.0.1",
    description="Predicting Loan Approval using machine learning",
    author="Chayan Halder",
    author_email="chayanh72@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)