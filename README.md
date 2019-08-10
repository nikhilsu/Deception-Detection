# Deception-Detection
A Machine learning model to detect deceptive(fake) Hotel and Electronic reviews

### Dataset Used:
[Boulder Lies and Truth dataset](https://catalog.ldc.upenn.edu/LDC2014T24)

### Model Architecture:

<img src="https://github.com/nikhilsu/Deception-Detection/raw/master/model.png" alt="Model Architecture" width="300"/>

### Project Dependencies:
- The project dependencies(python libraries) can be installed by running the following command:-
```bash
$ pip install -r requirements.txt
```


### Train model:
- Run the below commands to start training and evaluating the network.
    - You will need to provide the path to the dataset, and
    - A flag(*treat_F_as_deceptive*) that tell the program whether to treat the 'F' label in the dataset as *deceptive* or to treat it as a unique class while training.
        - More information - [Paper](https://pdfs.semanticscholar.org/2020/69b7beb1069fa653953867ef4c4b78663499.pdf?_ga=2.256976139.144500798.1565130137-276775829.1564163481).

```bash
$ python main.py --path_to_dataset "<path to the BLT dataset>" --treat_F_as_deceptive <True/False>
```

### References
- [A Tangled Web: The Faint Signals of Deception in Text - Boulder Lies and Truth Corpus (BLT-C)](http://www.lrec-conf.org/proceedings/lrec2016/summaries/1203.html)
- [V. Sandifer, Anna & Wilson, Casey & Olmsted, Aspen. (2017). Detection of fake online hotel reviews](https://www.researchgate.net/profile/Aspen_Olmsted/publication/325075174_Detection_of_fake_online_hotel_reviews/links/5b68a939299bf14c6d94f4b2/Detection-of-fake-online-hotel-reviews.pdf)
- [A. Mukherjee, V. Venkataraman, B. Liu and N. Glance, "Fake Review Detection: Classification and Analysis of Real and Pseudo Reviews](http://www2.cs.uh.edu/~arjun/papers/UIC-CS-TR-yelp-spam.pdf)
- [Automatic detection of deceptive opinions using automatically identified specific details Nikolai Vogler](https://pdfs.semanticscholar.org/c05d/42ded4f7423c785f50a06633679fd36b5ca5.pdf)
- [Sentence classification using Bi-LSTM](https://towardsdatascience.com/sentence-classification-using-bi-lstm-b74151ffa565)
- [From Word Embeddings To Document Distances](http://proceedings.mlr.press/v37/kusnerb15.pdf)
- [Gensim](https://radimrehurek.com/gensim/models/word2vec.html)
- [bidirectional LSTM + keras](https://www.kaggle.com/snlpnkj/bidirectional-lstm-keras)
- [Evaluate the Performance Of Deep Learning Models in Keras](https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/)
