# Dockerfile for project

# inherit from monai image
FROM projectmonai/monai:1.1.0

# create install folder inside container
RUN mkdir /requirements_installation

# add requirements file to container
ADD requirements.txt /requirements_installation

# update to latest pip version 
RUN pip install -U pip

# check python version --> Python 3.8.13
RUN python --version

# install the required python packages
RUN pip install -r /requirements_installation/requirements.txt

# update libraries
RUN apt-get update 

# install plastimatch
RUN apt-get install plastimatch -y --fix-missing

# install htop
RUN apt-get install htop

