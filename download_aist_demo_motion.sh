#/bin/bash
base_path=$(pwd)
mkdir data && cd data
mkdir test && cd test

echo Downloading test motion sequences...
wget https://scanimate.is.tue.mpg.de/media/upload/demo_data/aist_demo_seq.zip
unzip aist_demo_seq.zip -d ./
rm aist_demo_seq.zip

cd $base_path
echo Done!