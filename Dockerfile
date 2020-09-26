# Starting from the official CNTK docker image (based on
# Ubuntu-16.04)
FROM nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04

# Install packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ gfortran \
	openmpi-bin \
	libsndfile-dev \
        software-properties-common \
	emacs && \
    rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.20 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libmpi.so.20.10.1 /usr/lib/x86_64-linux-gnu/libmpi.so.12 && \
    ldconfig
    
# Install Python 3.6.
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y python3.6 python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0

# Install Python packages.
RUN pip3 install --upgrade pip && \
    pip3 install numpy scipy librosa joblib webrtcvad wurlitzer cntk-gpu 


# Copy the repository inside the docker in /dihard18
WORKDIR /dihard18
COPY . .

# Install model.
RUN ./install_model.sh

# Make the eval script executable
RUN chmod +x ./run_eval.sh
