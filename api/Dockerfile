FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install python3.6 python3-pip -y

WORKDIR /workdir

COPY ./setup.py /workdir/setup.py
COPY ./requirements.txt /workdir/requirements.txt
RUN pip3 install -r requirements.txt

<<<<<<< HEAD
ENV TZ=Europe/Paris

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
#for opencv
RUN apt-get install ffmpeg libsm6 libxext6 -y
=======
#for opencv
RUN apt-get install ffmpeg libsm6 libxext6 -y
>>>>>>> 1f2d9ec4d7230158d8cfba4df452c9293a156712
