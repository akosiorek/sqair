#!/usr/bin/env bash

fileid="1JtCo32VTQaPyGyKkoUXi7nzx1kTgx1ya"
filename="sqair_models.tar.gz"
cookie=/tmp/cookie_$fileid

curl -c $cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb $cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' $cookie`&id=${fileid}" -o ${filename}

tar -xvf sqair_models.tar.gz
rm sqair_models.tar.gz
rm $cookie
