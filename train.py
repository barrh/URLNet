import datetime
import json
import pickle

from tqdm import tqdm
import pandas as pd

from file_utils import read_parquet
from TextCNN import *
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

def train_dev_step(x, y, emb_mode, is_train=True):
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
    else:
        step, loss, acc = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
    return step, loss, acc


def make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, batch_size, nb_epochs, shuffle=False):
    if FLAGS["model"]["emb_mode"] == 1:
        batch_data = list(zip(x_train_char_seq, y_train))
    elif FLAGS["model"]["emb_mode"] == 2:
        batch_data = list(zip(x_train_word, y_train))
    elif FLAGS["model"]["emb_mode"] == 3:
        batch_data = list(zip(x_train_char_seq, x_train_word, y_train))
    elif FLAGS["model"]["emb_mode"] == 4:
        batch_data = list(zip(x_train_char, x_train_word, y_train))
    elif FLAGS["model"]["emb_mode"] == 5:
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
    if FLAGS["model"]["emb_mode"] == 1:
        x_char_seq, y_batch = zip(*batch)
    elif FLAGS["model"]["emb_mode"] == 2:
        x_word, y_batch = zip(*batch)
    elif FLAGS["model"]["emb_mode"] == 3:
        x_char_seq, x_word, y_batch = zip(*batch)
    elif FLAGS["model"]["emb_mode"] == 4:
        x_char, x_word, y_batch = zip(*batch)
    elif FLAGS["model"]["emb_mode"] == 5:
        x_char, x_word, x_char_seq, y_batch = zip(*batch)

    x_batch = []
    if FLAGS["model"]["emb_mode"] in [1, 3, 5]:
        x_char_seq = pad_seq_in_word(x_char_seq, FLAGS["data"]["max_len_chars"])
        x_batch.append(x_char_seq)
    if FLAGS["model"]["emb_mode"] in [2, 3, 4, 5]:
        x_word = pad_seq_in_word(x_word, FLAGS["data"]["max_len_words"])
        x_batch.append(x_word)
    if FLAGS["model"]["emb_mode"] in [4, 5]:
        x_char, x_char_pad_idx = pad_seq(x_char, FLAGS["data"]["max_len_words"], FLAGS["data"]["max_len_subwords"],
                                         FLAGS["model"]["emb_dim"])
        x_batch.extend([x_char, x_char_pad_idx])
    return x_batch, y_batch

if __name__=="__main__":

    with open("configs/config.json", "r") as f_p:
        FLAGS = json.load(f_p)

    for key, val in FLAGS.items():
        print("{}={}".format(key, val))

    # urls, labels = read_data(FLAGS["data.data_dir"])
    top_1M_websites = pd.read_csv(FLAGS["data"]["alexa_ranking"], header=None)
    tokenizer = create_tokenizer_from_alexa(FLAGS["data"]["max_len_words"], top_1M_websites[1].to_list())
    high_freq_words = None
    debug = False
    data_we_train_on = "debug" if debug else "training_demo"
    ds = read_parquet(FLAGS["data"]['data_dir'], split=["debug"] if debug else ["training_demo", "test_demo", "validation_demo"])
    train_urls_ = ds[data_we_train_on]["normalized_url"].to_list()

    ngrams_dict, worded_id_x, words_dict, ngrams_dict, x_train_char, x_train_word, x_train_char_seq, y_train = get_features_for_data(
        train_urls_,
        ds[data_we_train_on]["label"].to_numpy(),
        tokenizer, high_freq_words, FLAGS)

    if not debug:
        all_data = {}
        for datatype in ["test_demo", "validation_demo"]:
            urls_ = ds[datatype]["normalized_url"].to_list()
            labels_ = ds[datatype]["label"].to_numpy()
            all_data[datatype] = get_features_for_data(
                urls_,
                labels_,
                tokenizer, high_freq_words, FLAGS)

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=session_conf)

        with sess.as_default():
            cnn = TextCNN(
                char_ngram_vocab_size=len(ngrams_dict) + 1,
                word_ngram_vocab_size=len(words_dict) + 1,
                char_vocab_size=len(ngrams_dict) + 1,
                embedding_size=FLAGS["model"]["emb_dim"],
                word_seq_len=FLAGS["data"]["max_len_words"],
                char_seq_len=FLAGS["data"]["max_len_chars"],
                l2_reg_lambda=FLAGS["train"]["l2_reg_lambda"],
                mode=FLAGS["model"]["emb_mode"],
                filter_sizes=list(map(int, FLAGS["model"]["filter_sizes"].split(","))))

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS["train"]["lr"])
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            print("Writing to {}\n".format(FLAGS["log"]["output_dir"]))
            if not os.path.exists(FLAGS["log"]["output_dir"]):
                os.makedirs(FLAGS["log"]["output_dir"])

            # Save dictionary files
            ngrams_dict_dir = FLAGS["log"]["output_dir"] + "subwords_dict.p"
            pickle.dump(ngrams_dict, open(ngrams_dict_dir, "wb"))
            words_dict_dir = FLAGS["log"]["output_dir"] + "words_dict.p"
            pickle.dump(words_dict, open(words_dict_dir, "wb"))

            # Save training and validation logs
            train_log_dir = FLAGS["log"]["output_dir"] + "train_logs.csv"
            with open(train_log_dir, "w") as f:
                f.write("step,time,loss,acc\n")
            val_log_dir = FLAGS["log"]["output_dir"] + "val_logs.csv"
            with open(val_log_dir, "w") as f:
                f.write("step,time,loss,acc\n")

            # Save model checkpoints
            checkpoint_dir = FLAGS["log"]["output_dir"] + "checkpoints/"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_prefix = checkpoint_dir + "model"
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)

            sess.run(tf.compat.v1.global_variables_initializer())

            train_batches, nb_batches_per_epoch, nb_batches = make_batches(x_train_char_seq, x_train_word, x_train_char,
                                                                           y_train, FLAGS["train"]["batch_size"],
                                                                           FLAGS['train']['nb_epochs'], True)

            min_dev_loss = float('Inf')
            dev_loss = float('Inf')
            dev_acc = 0.0
            print("Number of baches in total: {}".format(nb_batches))
            print("Number of batches per epoch: {}".format(nb_batches_per_epoch))

            it = tqdm(range(nb_batches), desc="emb_mode {} delimit_mode {} train_size {}".format(FLAGS["model"]["emb_mode"],
                                                                                                 FLAGS["data"]["delimit_mode"],
                                                                                                 y_train.shape[0]), ncols=0)
            for idx in it:
                batch = next(train_batches)
                x_batch, y_batch = prep_batches(batch)
                step, loss, acc = train_dev_step(x_batch, y_batch, emb_mode=FLAGS["model"]["emb_mode"], is_train=True)
                if step % FLAGS["log"]["print_every"] == 0:
                    with open(train_log_dir, "a") as f:
                        f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), loss, acc))
                    it.set_postfix(
                        trn_loss='{:.3e}'.format(loss),
                        trn_acc='{:.3e}'.format(acc),
                        dev_loss='{:.3e}'.format(dev_loss),
                        dev_acc='{:.3e}'.format(dev_acc),
                        min_dev_loss='{:.3e}'.format(min_dev_loss))
                if step % FLAGS["log"]["eval_every"] == 0 or idx == (nb_batches - 1):
                    total_loss = 0
                    nb_corrects = 0
                    nb_instances = 0
                    #ngrams_dict, worded_id_x, words_dict, ngrams_dict, x_train_char, x_train_word, x_train_char_seq, y_train
                    #        for datatype in ["test_demo", "validation_demo"]:

                    #all_data[datatype]
                    test_batches = make_batches(all_data["validation_demo"][6], all_data["validation_demo"][5],
                                                all_data["validation_demo"][4],
                                                all_data["validation_demo"][7],
                                                FLAGS['train']['batch_size'], 1, False)
                    for test_batch in test_batches:
                        x_test_batch, y_test_batch = prep_batches(test_batch)
                        step, batch_dev_loss, batch_dev_acc = train_dev_step(x_test_batch, y_test_batch,
                                                                             emb_mode=FLAGS["model"]["emb_mode"],
                                                                             is_train=False)
                        nb_instances += x_test_batch[0].shape[0]
                        total_loss += batch_dev_loss * x_test_batch[0].shape[0]
                        nb_corrects += batch_dev_acc * x_test_batch[0].shape[0]

                    dev_loss = total_loss / nb_instances
                    dev_acc = nb_corrects / nb_instances
                    with open(val_log_dir, "a") as f:
                        f.write(
                            "{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), dev_loss, dev_acc))
                    if step % FLAGS["log"]["checkpoint_every"] == 0 or idx == (nb_batches - 1):
                        if dev_loss < min_dev_loss:
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            min_dev_loss = dev_loss
