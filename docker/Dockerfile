FROM ubuntu

MAINTAINER Nicholas Bianco

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula \
        select true | debconf-set-selections

# Set DEBIAN_FRONTEND to avoid interactive timezone prompt when installing
# packages.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git \
        build-essential libtool autoconf \
        cmake \
        gfortran \
        wget \
        pkg-config \
        libopenblas-dev \
        liblapack-dev \
        libeigen3-dev \
        libcolpack-dev \
        libspdlog-dev \
        libadolc-dev \
        libdocopt-dev \
        python3 \
        python3-dev \ 
        python3-numpy \
        python3-scipy \
        python3-matplotlib python3-opencv \
        python3-setuptools \
        python3-doit \
        python3-pandas \
        python3-opencv \
        python3-pip \
        ttf-mscorefonts-installer \
        swig \
        r-base
        # libssl-dev \
        # libfontconfig1-dev \
        # libcurl4-openssl-dev \
        # libharfbuzz-dev \
        # libfribidi-dev \
        # libfreetype6-dev \
        # libpng-dev \
        # libtiff5-dev \
        # libjpeg-dev \

RUN pip3 install PyYAML

# Matplotlib's default backend requires a DISPLAY / Xserver.
RUN mkdir -p ~/.config/matplotlib && \
    echo 'backend : Agg' >> ~/.config/matplotlib/matplotlibrc && \
    echo 'font.sans-serif : Arial, Helvetica, sans-serif' >> ~/.config/matplotlib/matplotlibrc

# Check out the earliest commit with all features needed to reproduce the results
RUN git clone https://github.com/opensim-org/opensim-core.git /opensim-core \
        && cd /opensim-core \
        && git checkout d9ebf9389fabfe9cceb8ab46122e90b69a07689e

RUN cd /opensim-core \
        && git submodule update --init \
        && mkdir ../opensim_dependencies_build \
        && cd ../opensim_dependencies_build \
        && cmake ../opensim-core/dependencies \
                -DOPENSIM_PYTHON_WRAPPING=on \
                -DSUPERBUILD_casadi=on \
                -DSUPERBUILD_eigen=on \
                -DSUPERBUILD_ipopt=on \
                -DSUPERBUILD_ezc3d=on \
                -DSUPERBUILD_colpack=on \
                -DSUPERBUILD_docopt=on \
                -DSUPERBUILD_simbody=on \
                -DSUPERBUILD_spdlog=on \
        && make --jobs 4 \
        && echo "/opensim_dependencies_install/adol-c/lib64" >> /etc/ld.so.conf.d/opensim.conf \
        && echo "/opensim_dependencies_install/ipopt/lib" >> /etc/ld.so.conf.d/opensim.conf \
        && ldconfig \
        && rm -r /opensim_dependencies_build

RUN cd / \
        && mkdir build \
        && cd build \
        && cmake ../opensim-core \
            -DOPENSIM_WITH_CASADI=on \
            -DOPENSIM_WITH_TROPTER=off \
            -DBUILD_PYTHON_WRAPPING=on \
            -DBUILD_TESTING=off \
            -DBUILD_EXAMPLES=off \
            -DOPENSIM_INSTALL_UNIX_FHS=off \
        && make --jobs 4 install \
        && echo "/opensim-core-install/sdk/lib" >> /etc/ld.so.conf.d/opensim.conf \
        && echo "/opensim-core-install/sdk/Simbody/lib" >> /etc/ld.so.conf.d/opensim.conf \
        && ldconfig \
        && rm -r /build

RUN cd /opensim-core-install/sdk/Python && python3 setup.py install

# Clone the repository with the simulation code and results
RUN git clone https://github.com/stanfordnmbl/balance-exo-sim.git /balance-exo-sim \
        && cd /balance-exo-sim \
        && git submodule init \
        && git submodule update

