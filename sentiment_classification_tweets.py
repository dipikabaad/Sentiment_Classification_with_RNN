import csv
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# FILPATH to tweets dataset
INPUTFILE_PATH = "data/twitter-airline-sentiment/Tweets.csv"

tweets = []
train_tweets =[]
test_tweets = []
sentiment_class = set()
tweet_sent_class= []
# NLTK stopwords
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def tokenizer(sentence):
    tokens = sentence.split(" ")
    #tokens = [token for token in tokens]
    #print(tokens)
    tokens = [porter.stem(token.lower()) for token in tokens if not token.lower() in stop_words]
    return tokens

i = 0
with open(INPUTFILE_PATH, 'r') as csvfile:
    tweetreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in tweetreader:
        if i == 0:
            i += 1
            continue

        tweets.append(tokenizer(row[10]))
        tweet_sent_class.append(row[1])
        sentiment_class.add(row[1])
        i += 1

class_dict = {}
for index, class_name in enumerate(sentiment_class):
    class_dict[class_name] = index

vocab = {}
vocab_index = 0
for tokens in tweets:
    for key, token in enumerate(tokens):
        #all_tokens.add(token)
        if token not in vocab:
            vocab[token] = vocab_index
            vocab_index += 1

#train test split
train_tweets = tweets[:9000]
test_tweets = tweets[9000:]

def map_word_vocab(sentence):
    idxs = [vocab[w] for w in sentence]
    return torch.tensor(idxs, dtype=torch.long)

def map_class(sentiment):
    classes = [0 for i in range(len(sentiment_class))]
    classes[class_dict[sentiment]] = 1
    return torch.tensor([class_dict[sentiment]], dtype=torch.long)

def prepare_sequence(sentence):
    # create the input feature vector
    input = map_word_vocab(sentence)
    return input

# Setting the Embedding and hidden size
EMBEDDING_DIM = 50
HIDDEN_DIM = 10

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, input_size)

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word, hidden):
        embeds = self.word_embeddings(word)

        combined = torch.cat((embeds.view(1, -1), hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


rnn = RNN(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), len(sentiment_class))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.001)



f1 = open(('data/output_tweetsent_loss.csv'), 'w+')
fieldnames = ["epoch","loss"]
writer1 = csv.DictWriter(f1, fieldnames=fieldnames)
writer1.writeheader()
all_losses = []
for epoch in range(30):  # again, normally you would NOT do 300 epochs, it is toy data
    if epoch % 5 == 0:
        print("Finnished epoch " + str(epoch / 30 * 100)  + "%")
    total_loss = 0
    for i in range(len(train_tweets)):
        sentence = train_tweets[i]
        sent_class = tweet_sent_class[i]

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        hidden = rnn.init_hidden()
        rnn.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence)
        target_class = map_class(sent_class)

        # Step 3. Run our forward pass.
        for i in range(len(sentence_in)):
            class_scores, hidden = rnn(sentence_in[i], hidden)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        #print(class_scores)
        loss = loss_function(class_scores, target_class)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    all_losses.append(total_loss)
    writer1.writerow({"epoch": epoch,  "loss": total_loss})
f1.close()




sentiment_class = list(sentiment_class)
f = open(('data/output_tweets_test.csv'), 'w+')
fieldnames = ["tweet_desc", "class_size", "sentiment_pred","sentiment_name","actual_class"]
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()

y_pred = []
y_actual = []
with torch.no_grad():
    for i in range(len(test_tweets)):
        sentence = test_tweets[i]
        sent_class = tweet_sent_class[9000+i]
        inputs = prepare_sequence(sentence)
        hidden = rnn.init_hidden()
        for i in range(len(inputs)):
            class_scores, hidden = rnn(inputs[i], hidden)
        # for word i. The predicted tag is the maximum scoring tag.

        y_pred.append(sentiment_class[((class_scores.max(dim=1)[1].numpy()))[0]])
        y_actual.append(str(sent_class))
        writer.writerow({'tweet_desc':" ".join(sentence), 'class_size': (str(class_scores.size()[0]) + " " + str(class_scores.size()[1])) , 'sentiment_pred':((class_scores.max(dim=1)[1].numpy()))[0], 'sentiment_name':sentiment_class[((class_scores.max(dim=1)[1].numpy()))[0]] ,'actual_class': str(sent_class)})

f.close()

print(all_losses)
print(sentiment_class)
print(confusion_matrix(y_actual, y_pred, labels=sentiment_class))
print(accuracy_score(y_actual, y_pred))


