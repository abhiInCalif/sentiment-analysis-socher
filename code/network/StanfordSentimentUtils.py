

def buildStanfordSentimentDataSet(basedir=None, dataset_sentences="datasetSentences.txt", dataset_split="datasetSplit.txt",
                                    sentiment_labels="sentiment_labels.txt"):    
    if basedir is None:
        # no basedir was provided, so provide a default
        basedir = "/Users/abhinavkhanna/Documents/Princeton/Independent_Work/python-scraper/lib/stanfordSentimentTreebank/"
    
    fd_sent = open(basedir + dataset_sentences, "r")
    next(fd_sent)
    fd_split = open(basedir + dataset_split, "r")
    next(fd_split)
    fd_senti = open(basedir + sentiment_labels, 'r')
    next(fd_senti)
    
    # build the dictionaries that you need here
    index_to_sentence = {}
    index_to_sentiment = {}
    index_to_set = {}
    training_set = []
    dev_set = []
    test_set = []
    
    for line in fd_sent:
        split_line = line.split("\t")
        # you have actual line data that you can read in here.
        index_to_sentence[int(split_line[0])] = split_line[1].rstrip()
    
    for line in fd_senti:
        split_line = line.split("|")
        index_to_sentiment[int(split_line[0])] = split_line[1].rstrip()
    
    for line in fd_split:
        split_line = line.split(",")
        index_to_set[int(split_line[0])] = int(split_line[1].rstrip())
    
    for k, v in index_to_sentence.items():
        set_index = index_to_set[k]
        if set_index == 1:
            training_set.append((v, index_to_sentiment[k]))
        if set_index == 2:
            test_set.append((v, index_to_sentiment[k]))
        if set_index == 3:
            dev_set.append((v, index_to_sentiment[k]))
    
    return training_set, test_set, dev_set
