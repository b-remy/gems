import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gems",
    version="0.0.1",
    author="Benjamin Remy",
    author_email="remy.benjamin.101@gmail.com",
    description="GEnerative Morphology for Shear",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/b-remy/gems",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #install_requires=['edward2[tensorflow]'],
    python_requires='>=3.6',
)
