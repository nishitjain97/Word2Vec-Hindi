# Word2Vec-Hindi
A skip-gram and CBOW Word2Vec model to create Hindi word embeddings on IIT Bombay Hindi monolingual corpus.

# Dataset
The dataset has been downloaded from the following link:

http://www.cfilt.iitb.ac.in/iitb_parallel/

Anoop Kunchukuttan, Pratik Mehta, Pushpak Bhattacharyya. The IIT Bombay English-Hindi Parallel Corpus. Language Resources and Evaluation Conference. 2018.

The details of the dataset are as follows:

| Item | Count |
| --- | --- |
| Lines of text | 5000000|
| Words in text | 81120156 |
| Vocabulary size | 843415 |
| Embedding size | 128 |

## Skip-gram
In this model of Word2Vec, we use a single word from the text and try to predict it's context.
Eg. for a sentence 'i love icecreams', our input would be 'love' and we would train the model to predict 'i' and 'icecreams'.

Following is a TSNE plot of the final embeddings of randomly selected 400 Hindi words.
![TSNE plot of Skip-gram Word2Vec embeddings of 400 words](https://github.com/nishitjain97/Word2Vec-Hindi/blob/master/skip-gram-tsne-plot.png)

## Continuous Bag-of-Words
In this model of Word2Vec, we use the context words and try to predict our word from the context.
Eg. for a sentence 'i love icecreams', our input would be ['i', 'icecreams'] and we would train the model to predict 'love'.

Following is a TSNE plot of the final embeddings of randomly selected 400 Hindi words.
![TSNE plot of CBOW Word2Vec embeddings of 400 words](https://github.com/nishitjain97/Word2Vec-Hindi/blob/master/CBOW-tsne-plot.png)

## Trained Embeddings
### Skip-Gram
Skip-Gram embeddings Pickle file:

https://drive.google.com/file/d/1gM71iFPr1gCVOWbXJfg7e1FFD2h5fb-u/view?usp=sharing

The file is a dictionary object with keys:
  1. embeddings
  2. dictionary
  3. reverse_dictionary

Directions of use:

```python
import pickle

with open('embeddings_sg.hi', 'rb') as f:
    summary = pickle.load(f)

print(summary['embeddings'].shape)
print(summary['dictionary'].keys())
print(summary['reverse_dictionary'].keys())
```

### CBOW
CBOW embeddings Pickle file:

https://drive.google.com/open?id=1QTkV2mvKFQCSoMS7OUQkClPDle3MV_Nf

The file is a dictionary object with keys:
  1. embeddings
  2. dictionary
  3. reverse_dictionary
 
Directions of use:

```python
import pickle

with open('embeddings_cbow.hi, 'rb') as f:
    summary = pickle.load(f)
    
print(summary['embeddings'].shape)
print(summary['dictionary'].keys())
print(summary['reverse_dictionary'].keys())
```
