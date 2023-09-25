#!/bin/bash
fileid="1Im3y2X6iwhZHFJLIOKbIGtZ0ZHBWaqD_"
filename="data.zip"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
echo "Downloading data archive..."
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
echo "Unzipping archive..."
unzip data.zip -d .
rm ./data.zip
rm ./cookie
