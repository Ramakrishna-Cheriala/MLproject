from setuptools import find_packages, setup
from typing import List


def get_requirments(file_path: str) -> List[str]:
    requirments = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments = [i.replace("\n", "") for i in requirments]

        if "-e ." in requirments:
            requirments.remove("-e .")

    return requirments


setup(
    name="mlproject",
    version="0.0.1",
    author="Ramakrishna",
    author_email="ramakrishnacheriala@gmail.com",
    packages=find_packages(),
    install_requires=get_requirments("requirments.txt"),
)
