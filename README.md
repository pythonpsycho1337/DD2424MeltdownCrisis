# Sentiment Analysis
This repository contains our project for the DD2424 course at KTH, Spring 2018. In this project, sentiment analysis on Twitter and Movie Reveiews was conducted.
For more information see the report see the report.pdf in the documents directory.

This code was executed with(It might work for other versions but that is not guaranteed):
Python version 3.6.5  
GCC version 7.2.0  
distro==1.2.0  
docutils==0.14  
gast==0.2.0  
gensim==3.4.0  
google-compute-engine==2.8.1  
grpcio==1.11.0  
html5lib==0.9999999  
idna==2.6  
jmespath==0.9.3  
Markdown==2.6.11  
mkl-fft==1.0.2  
mkl-random==1.0.1  
numpy==1.14.2  
protobuf==3.5.2  
pycosat==0.6.3  
pycparser==2.18  
pyOpenSSL==17.5.0  
PySocks==1.6.8  
python-dateutil==2.7.2  
requests==2.18.4  
ruamel-yaml==0.15.35  
s3transfer==0.1.13  
scipy==1.0.1  
six==1.11.0  
smart-open==1.5.7  
tensorboard==1.7.0  
tensorflow==1.7.0  
tensorflow-tensorboard==1.5.1  
termcolor==1.1.0  
urllib3==1.22  
webencodings==0.5  
Werkzeug==0.14.1  

Operating systems: Ubuntu 16.04

Static architecture(Master branch):
    To generate the preprocessed datasets make sure that the data is located in the correct folder and that you have the wordvec.bin file. Then run "python generate.py" in the terminal.

    To run an experiment:
        1: Set the dataset variable in main.py ("MR" or "Twitter")
        2: Customize the for-loop in main.py
        3: run "python main.py" in the terminal

Non-static architecture (Found in non-static Branch):
    first choose between dataset "Twitter" or "MR" by setting the dataset variable in main.py
    To generate the initial word embedding call wordvec.save_word2vec_dictionary in main.py

    To run an experiment:
        1: Set the dataset variable in main.py ("MR" or "Twitter")
        2: Customize the parameters in Parameters.py
        3: run "python main.py" in the terminal
