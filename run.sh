wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz
tar -xvzf nsynth-test.jsonwav.tar.gz -C .
python3.5 preprocess.py
python3.5 train.py
