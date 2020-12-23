"""This script is WIP and may change significantly during development."""
import re
import logging
from pprint import pprint
import warnings

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import spacy

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(mallet=True, score=False):
    """Script wrapper to prevent multiprocessing runtime errors."""
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')

    # Convert to list and remove email addresses, new lines and single quotation marks.
    data = df.content.values.tolist()
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]

    def sent_to_words(sentences):
        """Tokenises a sentence in a list of sentences into a list of words,
        removing punctuation and unnecessary characters.

        Parameters
        ----------
        sentences :obj:`list`
            List of sentences to be tokenised.

        """
        for sentence in sentences:
            # deacc=True removes punctuation.
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models - NB higher threshold yield fewer phrases.
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Cut down memory consumption of `Phrases` by discarding model state.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        """Removes stop words from list of words."""
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        """Forms bigrams from text samples."""
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        """Forms trigrams from text samples."""
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """Group together inflected forms of the same word so they can be analysed
        together.

        Notes
        -----
        See https://spacy.io/api/annotation for further information.

        """
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(
                [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            )
        return texts_out

    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Create ID-frequency pairs for each word in document.
    data_lemmatized = lemmatization(
        data_words_bigrams,
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    )
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    # Human readable format of corpus (term-frequency)
    # print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

    if mallet:
        """Mallet's method is based on Gibb's sampling, which is a more accurate
        fitting method than variational Bayes, used in standard GenSim modelling.
        Requires mallet source code download (http://mallet.cs.umass.edu/). This
        is a Java package and so requires a JDK. Note this can only be used for
        demo's as the JDK requires a 'low-cost' commercial license (see
        https://docs.oracle.com/en/java/javase/index.html).
        """
        mallet_path = 'mallet-2.0.8/bin/mallet'
        ldamallet = gensim.models.wrappers.LdaMallet(
            mallet_path,
            corpus=corpus,
            num_topics=20,
            id2word=id2word
        )
        # Compute Coherence Score
        coherence_model_ldamallet = CoherenceModel(
            model=ldamallet,
            texts=data_lemmatized,
            dictionary=id2word,
            coherence='c_v'
        )
        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
        print('\nCoherence Score: ', coherence_ldamallet)
        pprint(ldamallet.show_topics(num_topics=-1, formatted=False))
    else:
        # Use the standard GenSim LDA model.
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=20,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        doc_lda = lda_model[corpus]
        pprint(lda_model.print_topics())

        if score:
            # Compute perplexity - how 'surprised' the model is at new data.
            # Compute topic coherence - measures how good the model is.
            print('\nPerplexity: ', lda_model.log_perplexity(corpus))
            coherence_model_lda = CoherenceModel(
                model=lda_model,
                texts=data_lemmatized,
                dictionary=id2word,
                coherence='c_v'
            )
            coherence_lda = coherence_model_lda.get_coherence()
            print('\nCoherence Score: ', coherence_lda)

    # Visualise using pyLDAvis. Returns html that can be opened in chrome.
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
    pyLDAvis.save_html(vis, 'lda.html')

    return


if __name__ == "__main__":
    main()
