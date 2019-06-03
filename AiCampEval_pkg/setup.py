import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='AiCampEval',  
    version='1.4',
    author="Alpheus Lim, Ivan Yang",
    author_email="dstadh17@gmail.com",
    description="Module to evaluate deep learning model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brainhack-dsta/til-ai-camp",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    install_requires=[
        "pillow",
        "numpy",
        "wget",
        "pprint"
    ],
 )
