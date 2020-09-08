#! /bin/bash

reference=${1}
embdir=${2}

for i in $embdir/*.model
do
    echo ${i}
    python3 align.py -e0 ${reference} -e1 ${i}
done
