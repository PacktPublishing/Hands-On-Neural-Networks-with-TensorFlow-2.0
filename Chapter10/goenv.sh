#!/usr/bin/env bash

# variables
TF_VERSION_MAJOR=1
TF_VERSION_MINOR=13
TF_VERSION_PATCH=1


curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-""$TF_VERSION_MAJOR"."$TF_VERSION_MINOR"."$TF_VERSION_PATCH"".tar.gz" | sudo tar -C /usr/local -xz
sudo ldconfig
git clone https://github.com/tensorflow/tensorflow $GOPATH/src/github.com/tensorflow/tensorflow/
pushd $GOPATH/src/github.com/tensorflow/tensorflow/tensorflow/go
git checkout r"$TF_VERSION_MAJOR"."$TF_VERSION_MINOR"
go build
