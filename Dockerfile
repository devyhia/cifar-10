FROM ubuntu:14.04

#3.4.3
ENV PYTHON_VERSION 2.7
ENV NUM_CORES 4

# Install OpenCV 3.0
RUN apt-get -y update
RUN apt-get -y install python$PYTHON_VERSION-dev wget unzip \
                       build-essential cmake git pkg-config libatlas-base-dev gfortran \
                       libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
                       libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install numpy matplotlib

RUN wget https://github.com/Itseez/opencv/archive/2.4.13.zip -O opencv.zip && \
    unzip -q opencv.zip && mv /opencv-2.4.13 /opencv

RUN mkdir /opencv/build
WORKDIR /opencv/build
RUN cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    ..
RUN make -j$NUM_CORES
RUN make install

RUN ldconfig

RUN pip install jupyter
RUN pip install notebook
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install scipy
RUN pip install seaborn
RUN pip install six
RUN pip install h5py

RUN apt-get -y install python-tk

EXPOSE 8888
