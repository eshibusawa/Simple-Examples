#!/bin/bash

# BSD 2-Clause License
#
# Copyright (c) 2024, Eijiro SHIBUSAWA
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

Help()
{
   # Display Help
   echo "Script to build docker continer for cuda"
   echo
   echo "-u     user name"
   echo "-i     user id"
   echo "-h     Help."
   echo
}

# set default
USERNAME_ARG=$(id -nu)
UID_ARG=$(id -u)

while getopts u:i:h flag
do
    case "${flag}" in
        u) USERNAME_ARG=${OPTARG};;
        i) UID_ARG=${OPTARG};;
        h) Help
            exit;;
    esac
done

read -p password?: -s PASSWORD
echo

docker build --quiet --build-arg USERNAME=$USERNAME_ARG --build-arg PASSWORD=$PASSWORD --build-arg UID=$UID_ARG \
  -t u2204c125 .
