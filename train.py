import datetime
import json
import pickle

from tqdm import tqdm
import pandas as pd

#from TextCNN import *
from TextCNN import TextCNN
from utils import *

def get_features_for_data(train_urls,labels_,tokenizer, high_freq_words, FLAGS):
    train_urls = [url.lower() for url in train_urls]
    train_urls = normalize_urls(train_urls)
    ngramed_id_x, ngrams_dict, worded_id_x, words_dict = urls_to_ngrams(tokenizer, train_urls, high_freq_words,
                                                                        FLAGS["data"]["max_len_words"],
                                                                        FLAGS["data"]["delimit_mode"],
                                                                        FLAGS["data"]["max_len_subwords"])
    chared_id_x = char_id_x(train_urls, ngrams_dict, FLAGS["data"]["max_len_chars"])

    pos_x_train = np.where(labels_)[0]
    neg_x_train = np.where(labels_ == 0)[0]

    print("Overall Mal/Ben split: {}/{}".format(len(pos_x_train), len(neg_x_train)))

    _, _, x_train, y_train = prep_train_test(pos_x_train, neg_x_train, 0)
    x_train_char = get_ngramed_id_x(x_train, ngramed_id_x)
    x_train_word = get_ngramed_id_x(x_train, worded_id_x)
    x_train_char_seq = get_ngramed_id_x(x_train, chared_id_x)
    return  ngrams_dict, worded_id_x, words_dict ,ngrams_dict, x_train_char,x_train_word,x_train_char_seq, y_train
# data args
# max_len_words - maximum length of url in words
# max_len_chars - maximum length of url in characters
# max_len_subwords - maxium length of word in subwords
# min_word_freq - minimum frequency of word in training population to build vocabulary
# delimit_mode - 0: delimit by special chars, 1: delimit by special chars + each char as a word
# model args 

# emb_mode - 1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(default_emb_mode))


# x, word_reverse_dict = get_word_vocab(train_urls, FLAGS["data.max_len_words"])


###################################### Training #########################################################

def train_step(x, y, emb_mode, is_train=True):
    if is_train:
        p = 0.5
    else:
        p = 1.0
    if emb_mode == 1:
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 2:
        feed_dict = {
            cnn.input_x_word: x[0],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 3:
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_x_word: x[1],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 4:
        feed_dict = {
            cnn.input_x_word: x[0],
            cnn.input_x_char: x[1],
            cnn.input_x_char_pad_idx: x[2],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 5:
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_x_word: x[1],
            cnn.input_x_char: x[2],
            cnn.input_x_char_pad_idx: x[3],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    if is_train:
        _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
    return step, loss, acc


def make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, batch_size, nb_epochs, shuffle=False):
    if args["model"]["emb_mode"] == 1:
        batch_data = list(zip(x_train_char_seq, y_train))
    elif args["model"]["emb_mode"] == 2:
        batch_data = list(zip(x_train_word, y_train))
    elif args["model"]["emb_mode"] == 3:
        batch_data = list(zip(x_train_char_seq, x_train_word, y_train))
    elif args["model"]["emb_mode"] == 4:
        batch_data = list(zip(x_train_char, x_train_word, y_train))
    elif args["model"]["emb_mode"] == 5:
        batch_data = list(zip(x_train_char, x_train_word, x_train_char_seq, y_train))
    batches = batch_iter(batch_data, batch_size, nb_epochs, shuffle)

    if nb_epochs > 1:
        nb_batches_per_epoch = int(len(batch_data) / batch_size)
        if len(batch_data) % batch_size != 0:
            nb_batches_per_epoch += 1
        nb_batches = int(nb_batches_per_epoch * nb_epochs)
        return batches, nb_batches_per_epoch, nb_batches
    else:
        return batches


def prep_batches(batch):
    if args["model"]["emb_mode"] == 1:
        x_char_seq, y_batch = zip(*batch)
    elif args["model"]["emb_mode"] == 2:
        x_word, y_batch = zip(*batch)
    elif args["model"]["emb_mode"] == 3:
        x_char_seq, x_word, y_batch = zip(*batch)
    elif args["model"]["emb_mode"] == 4:
        x_char, x_word, y_batch = zip(*batch)
    elif args["model"]["emb_mode"] == 5:
        x_char, x_word, x_char_seq, y_batch = zip(*batch)

    x_batch = []
    if args["model"]["emb_mode"] in [1, 3, 5]:
        x_char_seq = pad_seq_in_word(x_char_seq, args["data"]["max_len_chars"])
        x_batch.append(x_char_seq)
    if args["model"]["emb_mode"] in [2, 3, 4, 5]:
        x_word = pad_seq_in_word(x_word, args["data"]["max_len_words"])
        x_batch.append(x_word)
    if args["model"]["emb_mode"] in [4, 5]:
        x_char, x_char_pad_idx = pad_seq(x_char, args["data"]["max_len_words"], args["data"]["max_len_subwords"],
                                         args["model"]["emb_dim"])
        x_batch.extend([x_char, x_char_pad_idx])
    return x_batch, y_batch


if __name__=="__main__":
    with open("configs/config.json", "r") as f_p:
        args = json.load(f_p)

    for key, val in args.items():
        print("{}={}".format(key, val))

    # urls, labels = read_data(FLAGS["data.data_dir"])
    top_1M_websites = pd.read_csv(args["data"]["alexa_ranking"], header=None)
    tokenizer = create_tokenizer_from_alexa(args["data"]["max_len_words"], top_1M_websites[1].to_list())
    high_freq_words = None

    train_split, validation_split = [pd.read_parquet(os.path.join(args["data"]['data_dir'], 'debug_split_06_07_2022_11_32_17')),
                                     pd.read_parquet(os.path.join(args["data"]['data_dir'], 'debug_split_06_07_2022_11_32_17'))]
    train_urls_ = train_split["normalized_url"].to_list()

    ngrams_dict, worded_id_x, words_dict, ngrams_dict, x_train_char, x_train_word, x_train_char_seq, y_train = get_features_for_data(
        train_urls_,
        train_split["label"].to_numpy(),
        tokenizer, high_freq_words, args)

    cnn = TextCNN(
        char_ngram_vocab_size=len(ngrams_dict) + 1,
        word_ngram_vocab_size=len(words_dict) + 1,
        char_vocab_size=len(ngrams_dict) + 1,
        embedding_size=args["model"]["emb_dim"],
        word_seq_len=args["data"]["max_len_words"],
        char_seq_len=args["data"]["max_len_chars"],
        l2_reg_lambda=args["train"]["l2_reg_lambda"],
        mode=args["model"]["emb_mode"],
        filter_sizes=list(map(int, args["model"]["filter_sizes"].split(","))))

    train_batches, nb_batches_per_epoch, nb_batches = make_batches(x_train_char_seq, x_train_word, x_train_char,
                                                                   y_train, args["train"]["batch_size"],
                                                                   args['train']['nb_epochs'], True)

    min_dev_loss = float('Inf')
    dev_loss = float('Inf')
    dev_acc = 0.0
    print("Number of baches in total: {}".format(nb_batches))
    print("Number of batches per epoch: {}".format(nb_batches_per_epoch))

    it = tqdm(range(nb_batches), desc="emb_mode {} delimit_mode {} train_size {}".format(args["model"]["emb_mode"],
                                                                                         args["data"]["delimit_mode"],
                                                                                         y_train.shape[0]), ncols=0)
    for idx in it:
        batch = next(train_batches)
        x_batch, y_batch = prep_batches(batch)
        step, loss, acc = train_step(x_batch, y_batch, emb_mode=args["model"]["emb_mode"], is_train=True)
