#!/bin/bash

~/Documents/SOFTWARES/mmseqs/bin/mmseqs easy-cluster $1 clusterRes tmp --min-seq-id 0.5 -c 0.8 --cov-mode 1

