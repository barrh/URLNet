import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class TextCNN(pl.LightningModule):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size, \
        word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0, \
        filter_sizes=[3,4,5,6], mode=0, output_conv_channels = 256, dropout_keep_prob=0.5,lr=1e-3):
        super().__init__()
        self.mode = mode
        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.num_filters_total = output_conv_channels * len(filter_sizes)
        if mode in (4,5):
            self.embedded_x_char = nn.Embedding(char_ngram_vocab_size, embedding_size)
        if mode in (2,3,4,5):
            self.embedded_x_word = nn.Embedding(word_ngram_vocab_size, embedding_size)
        if mode in (1,3,5):
            self.embedded_x_char_seq = nn.Embedding(char_vocab_size, embedding_size, self.input_x_char_seq)
        if self.mode in (2,3):
            self.sum_ngram_x_expanded = torch.unsqueeze(self.embedded_x_word, -1)
        if self.mode in (2 ,3 ,4 ,5):
            self.conv_maxpool = [nn.Conv2d(embedding_size,output_conv_channels,filt) for filt in filter_sizes]


        if self.mode in (1 ,3 ,5):
            self.char_conv_maxpool_ = [nn.Conv2d(embedding_size,output_conv_channels,filt) for filt in filter_sizes]
        if self.mode in (3, 5):
            self.word_char_concat = nn.Linear(self.num_filters_total, 512)
            self.char_char_concat = nn.Linear(self.num_filters_total, 512)
        output_layer_size = (1024,512,256,128,1)
        self.output_layer = [nn.Linear(in_, out_) for in_,out_ in zip(output_layer_size[:-1],output_layer_size[1:])]
        self.dropout_keep_prob = dropout_keep_prob

            #for i, filter_size in enumerate(filter_sizes):
            #    with tf.name_scope("conv_maxpool_%s" % filter_size):
            #        filter_shape = [filter_size, embedding_size, 1, 256]
            #        b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
            #       w = tf.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="w")
            #         conv = tf.nn.conv2d(
            #             self.sum_ngram_x_expanded,
            #             w,
            #             strides = [1,1,1,1],
            #             padding = "VALID",
            #             name="conv")
            #         h = tf.F.relu(tf.nn.bias_add(conv,b), name="relu")
            #         pooled = tf.nn.max_pool(
            #             h,
            #             ksize=[1, word_seq_len - filter_size + 1, 1, 1],
            #             strides=[1,1,1,1],
            #             padding="VALID",
            #             name="pool")
            #         pooled_x.append(pooled)
            #
    def forward(self,x):
        if self.mode in (4,5):
            input_x_char = x["input_x_char"]
            input_x_char_pad_idx = x["input_x_char_pad_idx"]
            #self.input_x_char = tf.compat.v1.placeholder(tf.int32, [None, None, None], name="input_x_char")
            #self.input_x_char_pad_idx = tf.compat.v1.placeholder(tf.float32, [None, None, None, embedding_size], name="input_x_char_pad_idx")
        if self.mode in (2,3,4,5):
            #self.input_x_word = tf.compat.v1.placeholder(tf.int32, [None, None], name="input_x_word")
            input_x_word = x["input_x_word"]
        if self.mode in (1,3,5):
            #self.input_x_char_seq = tf.compat.v1.placeholder(tf.int32, [None, None], name="input_x_char_seq")
            input_x_char_seq = x["input_x_char_seq"]


        l2_loss = 0

        if self.mode in (4,5):
            #self.embedded_x_char = tf.nn.embedding_lookup(self.char_w, self.input_x_char)
            embedded_x_char = torch.mul(self.embedded_x_charm,input_x_char_seq)


            #self.embedded_x_char = tf.multiply(self.embedded_x_char, self.input_x_char_pad_idx)
        #if mode == 2 or mode == 3 or mode == 4 or mode == 5:
        #    self.embedded_x_word = tf.nn.embedding_lookup(self.word_w, self.input_x_word)
        if self.mode in (1 ,3 ,5):
            embedded_x_char_seq = self.embedded_x_char_seq(input_x_char_seq)
            #self.embedded_x_char_seq = tf.nn.embedding_lookup(self.char_seq_w, self.input_x_char_seq)

        if self.mode in (4, 5):
            #self.sum_ngram_x_char = nn.reduce_sum(self.embedded_x_char, 2)
            sum_ngram_x_char = torch.sum(embedded_x_char,2)
            sum_ngram_x = torch.sum([sum_ngram_x_char, self.embedded_x_word])
            #self.sum_ngram_x = tf.add(self.sum_ngram_x_char, self.embedded_x_word)

        if self.mode in (4,5):
            sum_ngram_x_expanded = torch.unsqueeze(sum_ngram_x,-1)

        if self.mode in (1,3,5):

            char_x_expanded = torch.unsqueeze(embedded_x_char_seq, -1)
    ########################### WORD CONVOLUTION LAYER ################################
        if self.mode in (4, 5):
            pooled_x =  [nn.MaxPool2d(F.relu(pool_(sum_ngram_x_expanded)),self.word_seq_len - pool_.kernel_size + 1) for pool_ in self.conv_maxpool]
            x_flat = torch.reshape(torch.concat(pooled_x,3),[-1,self.num_filters_total])
            h_drop_sum_ngram = nn.Dropout(self.dropout_keep_prob)(x_flat)


        ########################### CHAR CONVOLUTION LAYER ###########################
        if self.mode in (1, 3, 5):
            pooled_char_x =  [nn.MaxPool2d(F.relu(pool_(char_x_expanded)),self.char_seq_len - pool_.kernel_size + 1) for pool_ in self.char_conv_maxpool_]
            x_flat = torch.reshape(torch.concat(pooled_char_x, 3), [-1, self.num_filters_total])
            h_drop_char = nn.Dropout(self.dropout_keep_prob)(x_flat)
        
        ############################### CONCAT WORD AND CHAR BRANCH ############################
        if self.mode in (3 ,5):


            word_output = self.word_char_concat(h_drop_sum_ngram)
            char_output = self.char_char_concat(h_drop_char)

            conv_output = torch.concat([word_output, char_output], 1)
        elif self.mode in (2,4):
            conv_output = h_drop_sum_ngram
        elif self.mode == 1:
            conv_output = h_drop_char
        for i,layer in enumerate(self.output_layer):
            if i==0:
                output_ = F.relu(layer(conv_output))
            elif i<len(self.output_layer)-2:
                output_ = F.relu(layer(output_))
            else:
                output_ = layer(output_)
        return output_

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('cross entropy train', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('cross entropy val', loss)
        output_after_threshold = F.threshold(y_hat,0)
        tp = torch.sum(output_after_threshold & y)
        fp = torch.sum((~output_after_threshold) & y)
        fn = torch.sum(output_after_threshold & (~y))
        self.log("val precision", tp/(tp+fp))
        self.log("val recall", tp / (tp + fn))

        return loss
# data
if __name__=="__main__":
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=32)
    val_loader = DataLoader(mnist_val, batch_size=32)

    # model
    model = TextCNN()

    # training
    trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
    trainer.fit(model, train_loader, val_loader)

