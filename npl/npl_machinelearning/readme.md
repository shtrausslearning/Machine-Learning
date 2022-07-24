
### 1 | Introduction

- **Word Embeddings** Foundational knowledge needed for further textual analysis using NLP techniques
- **Language Model** progresses from the basics of the language model in NLP to model application and analysis
- **Text Classification** teaches us the techniques of text classification in NLP and gives an overview of sentiment analysis
- **Seq2Seq Model** teaches us about Seq2Seq modeling in detail and provides takeaways on model improvement

#### Vocabulary 

- The <code>tokeniser</code> automatically converts each vocabulary word to an integer ID
- This allows the tokenised sequences to be used in NLP algorithms 
- the <code>texts_to_sequences</code> function converts each vocabulary word in new_texts to its corresponding integer ID
- <code>fit_on_texts</code> is also included in the word_index, we can also fit on new_texts, but we wont have other unique word indicies from text_corpus 


```python
import tensorflow.keras.preprocessing as p

# fit_on_texts - initialise the object with a text corpus
# texts_to_sequences - convert pieces of text into sequences of tokens

# Fit on data
tokeniser = p.text.Tokenizer()
text_corpus = ['bob ate apples, and pears', 'fred ate apples!']
tokeniser.fit_on_texts(text_corpus)

new_texts = ['bob ate pears', 'fred ate pears']
print(tokeniser.texts_to_sequences(new_texts))
print(tokeniser.word_index)
```

```
[[3, 1, 5], [6, 1, 5]]
{'ate': 1, 'apples': 2, 'bob': 3, 'and': 4, 'pears': 5, 'fred': 6}
```

- Tokiniser (<code>p.text.Tokenizer()</code>) filters out any punctuation and white space
- When a new text (new_texts) contains words which are not in the corpus vocabulary (text_corpus) (known as out-of-vocabulary (OOV) words)

```python

tokeniser = p.text.Tokenizer(oov_token='OOV')
text_corpus = ['bob ate apples', 'fred ate apples!']
tokeniser.fit_on_texts(text_corpus)

new_texts = ['bob ate pears', 'fred ate pears']
print(tokeniser.texts_to_sequences(new_texts))
print(tokeniser.word_index)
```

```
[[4, 2, 1], [5, 2, 1]]
{'OOV': 1, 'ate': 2, 'apples': 3, 'bob': 4, 'fred': 5}
```

- <code>num_words</code> maximum number of vocabulary words to use. 
- If we set num_words=10 when initializing the tokeniser:
  - it will only use the 10 most frequent words in the vocabulary & filter out the remaining words
