wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
mkdir data
tar -xvzf nsynth-test.jsonwav.tar.gz -C ./data
python3 preprocess.py ./data/nsynth/train/audio
python3 train.py ./data_dir/