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


def remove_stopwords(texts, stop_words):
    """Removes stop words from list of words.

    Parameters
    ----------
    stop_words :obj:`list`
        List of words to be removed.

    Returns
    -------
    :obj:`list`
        Filtered list of words.

    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts, bigram_mod):
    """Forms bigrams from text samples.

    Parameters
    ----------

    Returns
    -------

    """
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, trigram_mod):
    """Forms trigrams from text samples.

    Parameters
    ----------

    Returns
    -------

    """
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Group together inflected forms of the same word so they can be analysed
    together.

    Parameters
    ----------

    Returns
    -------

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


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """Compute c_v coherence sensitivity to topic number.

    Parameters
    ----------
    dictionary :obj:`dict`
        Gensim dictionary.
    corpus :
        Gensim corpus
    texts :obj:`list`
        List of input texts.
    limit :obj:`int`
        Max number of topics.

    Returns
    -------
    :obj:`list`
        List of LDA topic models.
    :obj:`list`
        Coherence values corresponding to the LDA model with respective
        number of topics.

    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(
            mallet_path,
            corpus=corpus,
            num_topics=num_topics,
            id2word=id2word
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def format_topics_sentences(ldamodel, corpus, texts):
    """Returns dataframe breaking down most prevalent topic for each document.

    Parameters
    ----------

    Returns
    -------
    :obj:`pandas.DataFrame`
        Dataframe showing dominant topic, topic percentage contribution, topic
        keywords and the raw text for each document.

    """
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True
                )
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


def main(mallet=True, score=False):
    """Script wrapper to prevent multiprocessing runtime errors."""

    # Data preprocessing.

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')

    # Convert to list and remove email addresses, new lines and single quotation marks.
    data = df.content.values.tolist()
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models - NB higher threshold yield fewer phrases.
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Cut down memory consumption of `Phrases` by discarding model state.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_words_nostops = remove_stopwords(data_words, stop_words)
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Create ID-frequency pairs for each word in document.
    data_lemmatized = lemmatization(
        data_words_bigrams,
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'],
        nlp=nlp
    )
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    # Human readable format of corpus (term-frequency)
    # print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

    # Instantiate LDA model.

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

    # # Visualise using pyLDAvis. Returns html that can be opened in chrome.
    # vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
    # pyLDAvis.save_html(vis, 'lda.html')

    # The following segment computes coherence across a range of values.
    # limit = 15
    # start = 13
    # step = 1
    # model_list, coherence_values = compute_coherence_values(
    #     dictionary=id2word,
    #     corpus=corpus,
    #     texts=data_lemmatized,
    #     start=start,
    #     limit=limit,
    #     step=step
    # )
    # x = range(start, limit, step)
    # plt.plot(x, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()

    # for m, cv in zip(x, coherence_values):
    #     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamallet, corpus=corpus, texts=data)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Find most representative document for each topic
    best_doc_per_topic_df = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        best_doc_per_topic_df = pd.concat([best_doc_per_topic_df,
                                           grp.sort_values(['Perc_Contribution'],
                                           ascending=[0]).head(1)], axis=0)
    best_doc_per_topic_df.reset_index(drop=True, inplace=True)
    best_doc_per_topic_df.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    best_doc_per_topic_df.to_excel("best_doc_per_topic.xlsx")

    # Tabulate the topic distribution across documents.
    topic_num_keywords = df_topic_sents_keywords[["Dominant_Topic", "Topic_Keywords"]].drop_duplicates()
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts() # Count docs per topic.
    topic_counts.name = "Count"
    print(topic_counts)
    topic_perc_docs = round(topic_counts/topic_counts.sum(), 4)
    topic_perc_docs.name = "Percentage_Documents"
    temp_topic_distribution_df = topic_num_keywords.join(topic_counts, on="Dominant_Topic")
    topic_distribution_df = temp_topic_distribution_df.join(topic_perc_docs, on="Dominant_Topic")
    topic_distribution_df.reset_index()
    print(topic_distribution_df)
    topic_distribution_df.to_excel("topic_distribution.xlsx")

    return


if __name__ == "__main__":
    main()
