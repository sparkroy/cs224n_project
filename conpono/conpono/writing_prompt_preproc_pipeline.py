# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Beam pipeline to convert WikiText103 to shareded TFRecords."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import numpy as np
import hashlib
import random
from absl import app
from absl import flags
# import apache_beam as beam
from bert import tokenization
# from create_pretrain_data.preprocessing_utils import convert_instance_to_tf_example
from create_pretrain_data.preprocessing_utils import create_instances_from_document
from create_pretrain_data.preprocessing_utils import create_paragraph_order_from_document
# import tensorflow as tf
import tensorflow.compat.v1 as tf

FORMAT_BINARY = "binary"
FORMAT_PARAGRAPH = "paragraph"

flags.DEFINE_string(
    "input_file", "../../data", "Path to raw input files."
    "Assumes the filenames wiki.{train|valid|test}.raw")
flags.DEFINE_string("output_file", None, "Output TF example file.")
flags.DEFINE_string("vocab_file", "vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("max_sent_length", 70, "Maximum sequence length.")
flags.DEFINE_integer("max_para_length", 30, "Maximum sequence length.")
flags.DEFINE_integer("random_seed", 12345, "A random seed")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_enum(
    "format", FORMAT_BINARY, [FORMAT_BINARY, FORMAT_PARAGRAPH],
    "Build a dataset of either binary order or paragraph reconstrucition")

FLAGS = flags.FLAGS


def read_file(filename):
  """Read the contents of filename (str) and split into documents by chapter."""

  all_stories = []
  tot = 0
  with tf.gfile.GFile(filename, "r") as reader:
    for line in reader:
      line = line.strip()
      if not line:
        continue
      tot += 1
      # if tot > 10:
      #   break # hard code for quick testing!
      all_stories.append(line)

  # Remove empty documents
  all_stories = [x for x in all_stories if x]
  return all_stories # a list of stories (paragrahs)


def split_line_by_sentences(line):
  # Put trailing period back but not on the last element
  # because that usually leads to double periods.
  sentences = [l + "." for l in line.split(" . ")]
  sentences[-1] = sentences[-1][:-1]
  return sentences


def preproc_doc(document):
  """Convert document to list of TF Examples for binary order classification.

  Args:
      document: a wikipedia article as a list of lines

  Returns:
      A list of tfexamples of binary orderings of pairs of sentences in the
      document. The tfexamples are serialized to string to be written directly
      to TFRecord.
  """

  # Each document is a list of lines
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case) # just use lower case?

  # set a random seed for reproducability
  # hash_object = hashlib.md5(document[0])
  rng = random.Random(1)

  # Each document is composed of a list of text lines. Each text line is a
  # paragraph. We split the line into sentences but keep the paragraph grouping.
  # The utility functions below expect the document to be split by paragraphs.
  list_of_paragraphs = []
  for line in document: # each line is a story
    line = tokenization.convert_to_unicode(line)
    line = line.replace(u"\u2018", "'").replace(u"\u2019", "'")
    sents = split_line_by_sentences(line)
    sent_tokens = [tokenizer.tokenize(sent) for sent in sents if sent] # list of words
    list_of_paragraphs.append(sent_tokens)

  # In case of any empty paragraphs, remove them.
  list_of_paragraphs = [x for x in list_of_paragraphs if len(x) >= 16] # list of list of token
  print("total valid stories", len(list_of_paragraphs))

  # Convert the list of paragraphs into TrainingInstance object
  # See preprocessing_utils.py for definition
  # if FLAGS.format == FORMAT_BINARY:
  #   instances = create_instances_from_document(list_of_paragraphs,
  #                                              FLAGS.max_seq_length, rng)
  # elif FLAGS.format == FORMAT_PARAGRAPH:
    # instances = create_paragraph_order_from_document(list_of_paragraphs,
    #                                                  FLAGS.max_seq_length, rng)

  # Convert token lists into ids and add any needed tokens and padding for BERT
  tf_examples = [
      convert_instance_to_tf_example(tokenizer, paragraph,
                                     FLAGS.max_sent_length, FLAGS.max_para_length)
      for paragraph in list_of_paragraphs
  ]

  # Serialize TFExample for writing to file.
  tf_examples = [example.SerializeToString() for example in tf_examples]

  return tf_examples

def create_bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def convert_instance_to_tf_example(tokenizer, sent_tokens, max_sent_length,
                                   max_para_length):
  """Convert a list of strings into a tf.Example."""

  input_ids_list = [
      tokenizer.convert_tokens_to_ids(tokens) for tokens in sent_tokens
  ]
  features = collections.OrderedDict()

  # pack or trim sentences to max_sent_length
  # pack paragraph to max_para_length
  sent_tensor = []
  for i in range(max_para_length):
    if i >= len(input_ids_list):
      sent_tensor.append([0] * max_sent_length)
    else:
      padded_ids = np.pad(
          input_ids_list[i], (0, max_sent_length),
          mode="constant")[:max_sent_length]
      sent_tensor.append(padded_ids)
  sent_tensor = np.ravel(np.stack(sent_tensor))
  sent_tensor = sent_tensor.astype(int)
  features["sents"] = create_int_feature(sent_tensor)

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example

def main(_):
  # If using Apache BEAM, execute runner here.
  for mode in ["train", "test", "val"]:
    test_files = FLAGS.input_file + "/{}.wp_target".format(mode)
    test_tfrecord = FLAGS.input_file + "/{}_target.tfrecord".format(mode)
    stories = read_file(test_files)
    serialized_examples = preproc_doc(stories)
    with tf.io.TFRecordWriter(test_tfrecord) as writer:
      for example in serialized_examples:
        writer.write(example)

if __name__ == "__main__":
  app.run(main)