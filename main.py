import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle
import os

stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)


try:
    with open('saveData.pickle', 'rb') as x:
        words, labels, training, output = pickle.load(x)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:

            # Get main word (IE: Whats -> What)
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))  # Remove Duplicates

    labels = sorted(labels)

    # Bag Of Words

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        listOfWords = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in listOfWords:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    out_empty = np.array(output)

    with open('saveData.pickle', 'wb') as x:
        pickle.dump((words, labels, training, output), x)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

if os.path.exists("ChatBotModel.tflearn.meta"):
    model.load('ChatBotModel.tflearn')
else:
    model.fit(training, output, n_epoch=1000,
              batch_size=8, show_metric=True)
    model.save('ChatBotModel.tflearn')


def baOfWords(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    print('Talk Here')
    while True:
        inp = input('Text: ')
        if inp.lower() == 'quit':
            break

        results = model.predict([baOfWords(inp, words)]).squeeze().squeeze()
        resultsIndex = np.argmax(results)
        tag = labels[resultsIndex]

        if results[resultsIndex] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print('I didn\'t understand')


chat()
