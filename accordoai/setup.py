from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="accordoai",
    version="0.2.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy",
        "librosa",
        "music21",
        "filetype",
        "pandas",
        "scipy",
        "numpy",
        "moviepy"
    ],
    author="Valenteno Lenora",
    author_email="valentenocavlenora@gmail.com",
    description="Chord prediction model from Accordo.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NightKing-V/Chord-Classification-Model-accordo.ai-",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
