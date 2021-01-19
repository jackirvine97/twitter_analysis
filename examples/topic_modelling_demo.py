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
import matplotlib.colors as mcolors
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import spacy
from wordcloud import WordCloud, STOPWORDS

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
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True
                )
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def plot_word_count_and_weight_per_topic(data_lemmatized, topic_keyword_wt):
    """Plots bar charts for each topic, displaying word count and weight.

    Parameters
    ----------
    ldamodel : gensim.models.ldamodel.LdaModel/gensim.models.wrappers.ldamallet.LdaMallet
        Gensim LDA model instantiated. Can be either a standard gensim model
        or gensim wrapper for LDA Mallet model.

    """
    # Plot word count and word weight per topic.
    data_flat = [w for w_list in data_lemmatized for w in w_list]
    counter = Counter(data_flat)
    out = []
    for i, topic in topic_keyword_wt:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(4, 5, figsize=(10, 7), sharey=True)
    cols = [cm.tab20(x) for x in range(20)]
    shuffle(cols)
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :],
               color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax.tick_params(axis='y', labelsize=3)
        ax_twin = ax.twinx()
        ax_twin.tick_params(axis='y', labelsize=3)
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :],
                    color=cols[i], width=0.2, label='Weights')
        ax_twin.set_ylim(0, 0.125)
        ax.set_ylabel('Word Count', color=cols[i], fontsize=3)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=6)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30,
                           horizontalalignment='right', fontsize=3)
        ax.legend(loc='upper left', fontsize=3)
        ax_twin.legend(loc='upper right', fontsize=3)

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=8, y=1.05)
    plt.show()

    return


def plot_document_count_per_topic(results_by_topic):
    """Plots bar charts showing number of documents by dominant topic and
    number of documents by topic 'weight' in the corpus.

    Parameters
    ----------
    results_by_topic : pandas.DataFrame
        Output from the INSERT FUNCTION NAME HERE. Intention is for this
        to be a standard dataframe native to this script.

    Notes
    -----
    Produces two plots. The first plots the number of documents in which a
    given topic is dominant. The second plots the topic weightage, a measure
    of the proportion of the corpus that belongs to each topic.

    """

    results_by_topic_sorted = results_by_topic.sort_values("Dominant_Topic_Count",
                                                           ascending=False)
    x_ticks = [f"Topic {topic_id}\n{words}" for topic_id, words in
               zip(results_by_topic_sorted.Topic_ID, results_by_topic_sorted.Top_3_Words)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=120, sharey=True)
    ax1.bar(
        x=x_ticks,
        height='Dominant_Topic_Count',
        data=results_by_topic_sorted[["Dominant_Topic_Count", "Top_3_Words", "Topic_ID"]],
        width=.5,
        color='firebrick'
    )
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=8))
    ax1.set_ylabel('Number of Documents', size=5)
    ax1.tick_params(labelsize=5)

    ax2.bar(
        x=x_ticks,
        height='Corpus_Weight_Count',
        data=results_by_topic_sorted[["Corpus_Weight_Count", "Top_3_Words", "Topic_ID"]],
        width=.5, color='steelblue'
    )
    ax2.set_title('Number of Documents by Corpus Weight', fontdict=dict(size=8))
    ax2.set_ylabel('Number of Documents', size=5)
    ax2.tick_params(labelsize=5)

    fig.tight_layout()
    plt.show()

    return


def topics_per_document(model, corpus, start=0, end=-1):
    """Returns dominant topic and topic contribution for every document.

    Parameters
    ----------
    ldamodel : gensim.models.ldamodel.LdaModel/gensim.models.wrappers.ldamallet.LdaMallet
        Gensim LDA model instantiated. Can be either a standard gensim model
        or gensim wrapper for LDA Mallet model.
    corpus : list
        Model corpus.
    start : int
        Lower corpus index to be extracted.Default is 0.
    end : int
        Upper corpus index to be extracted. Default is -1.

    Returns
    -------
    tuple
        Tuple containing two lists, giving the dominant topic for each
        document, and the topic contribution for to each document.

    """

    corpus_sel = corpus[start:end]
    doc_topic_weights = model[corpus_sel]
    dominant_topics = []
    topic_percentages = []
    topic_percentages = [topic_percs for topic_percs in doc_topic_weights]
    dominant_topics = [(i, sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0])
                       for i, topic_percs in enumerate(doc_topic_weights)]
    return(dominant_topics, topic_percentages)


def results_by_topic_df(lda_model, corpus, topic_keyword_wt, *, save_as_excel=None,
                        save_as_pickle=None):
    """Returns dataframe breaking down keywords and distribution per topic.

    Parameters
    ----------
    ldamodel : gensim.models.ldamodel.LdaModel/gensim.models.wrappers.ldamallet.LdaMallet
        Gensim LDA model instantiated. Can be either a standard gensim model
        or gensim wrapper for LDA Mallet model.
    corpus : list
        Model corpus.
    save_as_excel : bool/str
        If `True`, the results by topic dataframe is saved as an excel file
        in current working directory under a default name. If `str` specified,
        this is used as the excel file name/path.
    save_as_pickle : bool/str
        If `True`, the results by topic dataframe is saved as a pickle file
        in current working directory under a default name. If `str` specified,
        this is used as the pickle file name/path.

    Returns
    -------
    pd.DataFrame
        The results by topic dataframe.

    """

    dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus)

    # Topic distribution by dominance.
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='Dominant_Topic_Count')
    df_dominant_topic_in_each_doc.reset_index(inplace=True)
    df_dominant_topic_in_each_doc.rename(columns={"Dominant_Topic": "Topic_ID"},
                                         inplace=True)

    # Topic distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='Corpus_Weight_Count')
    df_topic_weightage_by_doc.reset_index(inplace=True, drop=True)

    # Top 3 Keywords per topic - keyword handling not efficient - needs fixed.
    topic_top3words = [(i, topic) for i, topics in topic_keyword_wt
                       for j, (topic, wt) in enumerate(topics) if j < 3]

    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['Topic', 'Top_3_Words'])
    df_top3words = df_top3words_stacked.groupby('Topic').agg(', \n'.join)
    df_top3words.reset_index(level=0, inplace=True)

    # Keywords for each Topic.
    topic_keywords = [(i, topic) for i, topics in topic_keyword_wt
                      for j, (topic, wt) in enumerate(topics) if j < 10]

    df_topic_keywords_stacked = pd.DataFrame(topic_keywords, columns=['Topic_', 'Keywords'])
    df_topic_keywords = df_topic_keywords_stacked.groupby('Topic_').agg(', \n'.join)
    df_topic_keywords.reset_index(level=0, inplace=True)

    results_by_topic = df_dominant_topic_in_each_doc.join(df_topic_weightage_by_doc, on="Topic_ID")
    results_by_topic = results_by_topic.join(df_top3words, on="Topic_ID")
    results_by_topic.drop("Topic", axis=1, inplace=True)
    results_by_topic = results_by_topic.join(df_topic_keywords, on="Topic_ID")
    results_by_topic.drop("Topic_", axis=1, inplace=True)

    if save_as_excel is not None:
        excel_filename = "Results_by_Topic.xlsx"
        if isinstance(save_as_excel, str):
            excel_filename = f"{save_as_excel}.xlsx"
        results_by_topic.to_excel(excel_filename)
    if save_as_pickle is not None:
        pickle_filename = "Results_by_Topic.pkl"
        if isinstance(save_as_excel, str):
            pickle_filename = f"{save_as_pickle}.pkl"
        results_by_topic.to_pickle(pickle_filename)

    return results_by_topic


def plot_in_pyldavis(lda_model, corpus, id2word):
    """Visualises using pyLDAvis. Returns html that can be opened in chrome.

    lda_model : gensim.models.ldamodel.LdaModel
        Gensim LDA model instantiated. If LDA mallet wrapper was used, the
        model must be converted back to a standard gensim lda model using
        `malletmodel2ldamodel` to ensure pyldavis compatibility.
    corpus: list
        Model corpus.
    id2word : dict

    Notes
    -----
    Import statements placed within function as a temporary measure to avoid
    deprecation warnings associated with importing the pyldavis package.

    """
    import pyLDAvis
    import pyLDAvis.gensim
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
    pyLDAvis.save_html(vis, 'lda.html')
    return


def plot_word_count_per_doc_histogram(dominant_topic_df):
    """ Plots word count per document histogram.

    Notes
    -----
    This currently produces a `UserWarning` related to the `FixedFormatter`.
    Fix is WIP.

    """
    doc_lens = [len(d) for d in dominant_topic_df.Text]

    plt.figure()
    plt.hist(doc_lens, bins=5000, color='navy')
    plt.text(5750, 150, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(5750, 140, "Median : " + str(round(np.median(doc_lens))))
    plt.text(5750, 130, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(5750, 120, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(5750, 110, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 8000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.title('Distribution of Document Word Counts')
    plt.show()

    # Plot word count histogram per topic with kernel density estimate (KDE).
    fig, axes = plt.subplots(4, 5, figsize=(8, 8), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = dominant_topic_df.loc[dominant_topic_df.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]

        ax.hist(doc_lens, bins=5000)
        ax.tick_params(axis='both', labelsize=3)
        ax.set_xlim(0, 4000)
        ax.set_xlabel('Document Word Count', fontsize=3)
        ax.set_ylabel('Number of Documents', fontsize=3)
        ax.set_title('Topic: '+str(i), fontsize=6)
        ax_2 = sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx(), lw=0.6)
        ax_2.tick_params(axis='y', labelsize=3)
        ax_2.set_ylim(0, 0.0008)
        ax_2.set_ylabel('Density', fontsize=3)
        ax_2.set_yticklabels(["{:.2e}".format(t) for t in ax_2.get_yticks()])

    fig.tight_layout()
    # fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=8)
    plt.show()

    return


def plot_t_sne_topic_clusters(lda_model, corpus, topic_keyword_wt):
    """Plot topic clusters using  t-distributed stochastic neighbour
    embedding (t-SNE).

    t-SNE is a non-linear dimensionality reduction algorithm used to reduce
    data from a high dimensional space to a lower lower of dimensions,
    rendering it suitable for plotting.

    ldamodel : gensim.models.ldamodel.LdaModel/gensim.models.wrappers.ldamallet.LdaMallet
        Gensim LDA model instantiated. Can be either a standard gensim model
        or gensim wrapper for LDA Mallet model.
    corpus : list
        Model corpus.

    """

    # Get topic weights and dominant topics.
    topic_weights = []
    for row_list in lda_model[corpus]:
        tmp = np.zeros(len(topic_keyword_wt))
        for i, w in row_list:
            tmp[i] = w
        topic_weights.append(tmp)
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    colours_list = [color for name, color in mcolors.CSS4_COLORS.items()]
    shuffle(colours_list)
    colours_array = np.array(colours_list)

    # Plot the Topic Clusters using Bokeh
    n_topics = len(topic_keyword_wt)
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=colours_array[topic_num])
    show(plot)
    return


def plot_topic_wordclouds(topic_keyword_wt, max_words, stop_words):
    """Plots wordclouds for most prevalent words in each topic distribution."""

    colours_list = [color for name, color in mcolors.CSS4_COLORS.items()]
    shuffle(colours_list)
    cloud = WordCloud(
        stopwords=stop_words,
        background_color='white',
        width=2500,
        height=1800,
        max_words=max_words,
        colormap='tab10',
        color_func=lambda *args, **kwargs: colours_list[i],
        prefer_horizontal=1.0
    )
    num_rows = math.ceil(len(topic_keyword_wt)/4)
    fig, axes = plt.subplots(num_rows, 4, figsize=(6.5, 6.5), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        if i >= len(topic_keyword_wt):
            break
        fig.add_subplot(ax)
        topic_words = dict(topic_keyword_wt[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=9))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    return


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

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency).
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

    if mallet:
        """Mallet's method is based on Gibb's sampling, which is a more accurate
        fitting method than variational Bayes, used in standard GenSim modelling.
        Requires mallet source code download (http://mallet.cs.umass.edu/). This
        is a Java package and so requires a JDK. Note this can only be used for
        demos as the JDK requires a 'low-cost' commercial license (see
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
        # Use the standard GenSim LDA model. This is currently not supported for
        # post processing or visualisation.
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

    # Find dominant topic for each document.
    df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamallet, corpus=corpus, texts=data)
    dominant_topic_df = df_topic_sents_keywords.reset_index()
    dominant_topic_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Find most representative document for each topic.
    best_doc_per_topic_df = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    for i, grp in sent_topics_outdf_grpd:
        best_doc_per_topic_df = pd.concat([best_doc_per_topic_df,
                                          grp.sort_values(['Perc_Contribution'],
                                           ascending=[0]).head(1)], axis=0)
    best_doc_per_topic_df.reset_index(drop=True, inplace=True)
    best_doc_per_topic_df.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    best_doc_per_topic_df.to_excel("best_doc_per_topic.xlsx")

    # Tabulate the dominant topic distribution across documents. Note this is different to the marginal
    # topic distribution charted in pyLDAvis (which is the % of words in the corpus a given topic covers).
    topic_num_keywords = df_topic_sents_keywords[["Dominant_Topic", "Topic_Keywords"]].drop_duplicates()
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()  # Count docs per topic.
    topic_counts.name = "Count"
    topic_perc_docs = round(topic_counts/topic_counts.sum(), 4)
    topic_perc_docs.name = "Percentage_Documents"
    temp_dominant_topic_distribution_df = topic_num_keywords.join(topic_counts, on="Dominant_Topic")
    dominant_topic_distribution_df = temp_dominant_topic_distribution_df.join(topic_perc_docs, on="Dominant_Topic")
    dominant_topic_distribution_df.reset_index(drop=True, inplace=True)
    dominant_topic_distribution_df.to_excel("dominant_topic_distribution.xlsx")

    # Plot wordclouds for the first 9 topics by index.
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    cloud = WordCloud(
        stopwords=stop_words,
        background_color='white',
        width=2500,
        height=1800,
        max_words=10,
        colormap='tab10',
        color_func=lambda *args, **kwargs: cols[i],
        prefer_horizontal=1.0
    )
    topics = ldamallet.show_topics(num_topics=9, formatted=False)

    fig, axes = plt.subplots(3, 3, figsize=(3, 3), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=11))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    main()
