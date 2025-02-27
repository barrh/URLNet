import collections
from six.moves import range
import six
import numpy as np


class Tokenizer:
    """Text tokenization utility class.
  Deprecated: `tf.keras.preprocessing.text.Tokenizer` does not operate on
  tensors and is not recommended for new code. Prefer
  `tf.keras.layers.TextVectorization` which provides equivalent functionality
  through a layer which accepts `tf.Tensor` input. See the
  [text loading tutorial](https://www.tensorflow.org/tutorials/load_data/text)
  for an overview of the layer and text handling in tensorflow.
  This class allows to vectorize a text corpus, by turning each
  text into either a sequence of integers (each integer being the index
  of a token in a dictionary) or into a vector where the coefficient
  for each token could be binary, based on word count, based on tf-idf...
  By default, all punctuation is removed, turning the texts into
  space-separated sequences of words
  (words maybe include the `'` character). These sequences are then
  split into lists of tokens. They will then be indexed or vectorized.
  `0` is a reserved index that won't be assigned to any word.
  Args:
      num_words: the maximum number of words to keep, based
          on word frequency. Only the most common `num_words-1` words will
          be kept.
      filters: a string where each element is a character that will be
          filtered from the texts. The default is all punctuation, plus
          tabs and line breaks, minus the `'` character.
      lower: boolean. Whether to convert the texts to lowercase.
      split: str. Separator for word splitting.
      char_level: if True, every character will be treated as a token.
      oov_token: if given, it will be added to word_index and used to
          replace out-of-vocabulary words during text_to_sequence calls
      analyzer: function. Custom analyzer to split the text.
          The default analyzer is text_to_word_sequence
  """

    def __init__(self,
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 char_level=False,
                 oov_token=None,
                 analyzer=None,
                 **kwargs):
        # Legacy support
        document_count = kwargs.pop('document_count', 0)
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        self.word_counts = collections.OrderedDict()
        self.word_docs = collections.defaultdict(int)
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = document_count
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = collections.defaultdict(int)
        self.word_index = {}
        self.index_word = {}
        self.analyzer = analyzer

    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.
    In the case where texts contains lists,
    we assume each entry of the lists to be a token.
    Required before using `texts_to_sequences` or `texts_to_matrix`.
    Args:
        texts: can be a list of strings,
            a generator of strings (for memory-efficiency),
            or a list of list of strings.
    """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                if self.analyzer is None:
                    seq = self.text_to_word_sequence(
                        text, filters=self.filters, lower=self.lower, split=self.split)
                else:
                    seq = self.analyzer(text)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def fit_on_sequences(self, sequences):
        """Updates internal vocabulary based on a list of sequences.
    Required before using `sequences_to_matrix`
    (if `fit_on_texts` was never called).
    Args:
        sequences: A list of sequence.
            A "sequence" is a list of integer word indices.
    """
        self.document_count += len(sequences)
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                self.index_docs[i] += 1

    @classmethod
    def text_to_word_sequence(cls, input_text,
                                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                  lower=True,
                                  split=' '):
            r"""Converts a text to a sequence of words (or tokens).
        Deprecated: `tf.keras.preprocessing.text.text_to_word_sequence` does not
        operate on tensors and is not recommended for new code. Prefer
        `tf.strings.regex_replace` and `tf.strings.split` which provide equivalent
        functionality and accept `tf.Tensor` input. For an overview of text handling
        in Tensorflow, see the [text loading tutorial]
        (https://www.tensorflow.org/tutorials/load_data/text).
        This function transforms a string of text into a list of words
        while ignoring `filters` which include punctuations by default.
        >>> sample_text = 'This is a sample sentence.'
        >>> tf.keras.preprocessing.text.text_to_word_sequence(sample_text)
        ['this', 'is', 'a', 'sample', 'sentence']
        Args:
            input_text: Input text (string).
            filters: list (or concatenation) of characters to filter out, such as
                punctuation. Default: ``'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n'``,
                  includes basic punctuation, tabs, and newlines.
            lower: boolean. Whether to convert the input to lowercase.
            split: str. Separator for word splitting.
        Returns:
            A list of words (or tokens).
        """
            if lower:
                input_text = input_text.lower()

            translate_dict = {c: split for c in filters}
            translate_map = str.maketrans(translate_dict)
            input_text = input_text.translate(translate_map)

            seq = input_text.split(split)
            return [i for i in seq if i]

    def texts_to_sequences(self, texts):
        """Transforms each text in texts to a sequence of integers.
        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        Args:
            texts: A list of texts (strings).
        Returns:
            A list of sequences.
        """
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` to a sequence of integers.
        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.
        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        Args:
            texts: A list of texts (strings).
        Yields:
            Yields individual sequences.
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                if self.analyzer is None:
                    seq = self.text_to_word_sequence(
                        text, filters=self.filters, lower=self.lower, split=self.split)
                else:
                    seq = self.analyzer(text)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x