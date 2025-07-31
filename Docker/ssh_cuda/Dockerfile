# BSD 2-Clause License
#
# Copyright (c) 2025, Eijiro SHIBUSAWA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG USERNAME=john-smith
ARG PASSWORD=password
ARG UID=1234

# set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# install ssh etc.
RUN apt clean && \
  apt update && \
  apt install -y tzdata openssh-server sudo git nano cmake wget rsync libgl1-mesa-dev
EXPOSE 22

# install vscode
RUN wget https://packages.microsoft.com/keys/microsoft.asc -O - | gpg --dearmor > /usr/share/keyrings/microsoft.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list
RUN apt update && \
  apt install -y code fonts-noto-cjk

# install cmake
RUN wget https://apt.kitware.com/keys/kitware-archive-latest.asc -O - 2>/dev/null | gpg --dearmor > /usr/share/keyrings/kitware-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" > /etc/apt/sources.list.d/kitware.list
RUN apt update && \
  apt install -y cmake

# add user
RUN useradd -rm -d /home/${USERNAME} -s /bin/bash -u ${UID} ${USERNAME}
RUN gpasswd -a ${USERNAME} sudo
RUN echo "${USERNAME}:${PASSWORD}" | chpasswd

## add user settings
USER ${USERNAME}
WORKDIR /home/${USERNAME}
RUN mkdir -p mnt
RUN echo '"\e[A":history-search-backward' > .inputrc
RUN echo '"\e[B":history-search-forward' >> .inputrc

##  install miniforge
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b -p /home/${USERNAME}/anaconda3
RUN ./anaconda3/bin/conda init
RUN rm -f Miniforge3-$(uname)-$(uname -m).sh

USER ${USERNAME}
