import tensorflow as tf
import tqdm
import hparams
import collections

hp = hparams.HParams().hparams
class DataLoader:
    def __init__(self, mode=None):
        self.in_train_path = r"./data/train/in.txt"
        self.out_train_path = r"./data/train/out.txt"
        self.in_test_path = r"./data/test/in.txt"
        self.out_test_path = r"./data/test/out.txt"
        self.vocab_path = r"./data/vocabs.txt"
        self.train_recode_path = r"./data/train.recoder"
        self.test_recode_path = r"./data/test.recoder"
        self.voacb_list = self.load_vocab()
        self.BEG_ID = 2
        self.EOF_ID = 3
        self.UNK_ID = 1
        self.PAD = 0
        self.mode = mode
        # in_id, out_id = self.load_data(mode)

    def generate_tf_file(self):
        in_ids, in_length_list, out_ids, ys = self.load_data(self.mode)

        if self.mode == "train":
            out_path = self.train_recode_path
        elif self.mode == "test":
            out_path = self.test_recode_path
        writer = tf.python_io.TFRecordWriter(out_path)

        for index, in_id in enumerate(in_ids):
            in_length = in_length_list[index]
            out_id = out_ids[index]
            y = ys[index]
            features = collections.OrderedDict()
            if self.mode == "train":
                features["in_ids"] = self._create_int_feature(in_id)
                features["length"] = self._create_int_feature([in_length])
                features["out_ids"] = self._create_int_feature(out_id)
                features["y"] = self._create_int_feature(y)
            elif self.mode == "test":
                features["in_ids"] = self._create_int_feature(in_id)
                features["length"] = self._create_int_feature([in_length])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()


    def load_data(self, mode="train"):
        """
        得到[[id1, id2, id3, ...], [id1, id2, id3, ...]....[....]]
        :param mode:
        :return:
        """
        if mode == "train":
            in_list = self.load_data_helper(self.in_train_path)
            out_list = self.load_data_helper(self.out_train_path)
            data_size_in = len(in_list)
            data_size_out = len(out_list)
            assert data_size_in == data_size_out

        elif mode == "test":
            in_list = self.load_data_helper(self.in_test_path)
            out_list = self.load_data_helper(self.out_test_path)
            data_size_in = len(in_list)
            data_size_out = len(out_list)
            assert data_size_in == data_size_out
        else:
            raise ValueError("Please check mode!")
        print("\nIN start to id ... 奥利给！！！")
        in_id_list, in_length_list = self.word2id(in_list, mode="x")

        print("\nOUT start to id ... 奥利给！！！")
        out_id_list, _ = self.word2id(out_list, mode="y_id")
        y_list, _ = self.word2id(out_list, mode="y")
        return in_id_list, in_length_list, out_id_list, y_list

    def load_data_helper(self, path):
        """
        返回 [[word1, word2, word3, ... ], [], [], ... [sentence_n]]
        :param path:
        :return:
        """
        res_list = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                res_list.append(line.split(" "))
        return res_list

    def word2id(self, sentence_list, mode):
        id_list = []
        length_list = []
        for sentence in tqdm.tqdm(sentence_list):
            sentence_id, length = self.word2id_help(sentence, mode)
            id_list.append(sentence_id)
            length_list.append(length)
        return id_list, length_list

    def word2id_help(self, sentence, mode="x"):
        """
        x [id_1, id_2, ... </s>]
        y_id [<s>, id1, id2 ....]
        y [id_1, id_2, ... </s>]
        :param sentence:
        :param mode:
        :return:
        """
        res_list = []
        if mode == "y_id":
            res_list.append(self.BEG_ID)  # beg id
        for word in sentence:
            id = self.voacb_list.index(word)
            if id > 9000:
                id = self.UNK_ID
            res_list.append(id)
        if mode != "y_id":
            res_list.append(self.EOF_ID)  # end id
        length = len(res_list)
        res_list.extend([self.PAD]*(hp.max_in_length-length))
        assert len(res_list) == hp.max_in_length
        return res_list, length

    def _create_int_feature(self, value):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))
        return f

    def load_vocab(self):
        vocab_list = []
        with open(self.vocab_path) as f:
            for line in f:
                line = line.strip()
                vocab_list.append(line)
        return vocab_list

def decoder_recoder(record, name_to_feature):
    """
    因为record数据中，数据以二进制的形式进行存储，所以需要对其进行解码
    :param record: 得到的recoder_dataset数据
    :param name_to_feature:解码的特征，格式为字典
    :return:example
    """
    example = tf.parse_single_example(record, name_to_feature)
    return example


def file_based_input_fn_builder(input_file_path, is_training, drop_remainder, mode):

    print("Input recoder_file from {} ...".format(input_file_path))

    def input_fn():
        if mode == "train":
            batch_size = hp.batch_size
        elif mode == "dev":
            batch_size = hp.eval_batch_size
        else:
            batch_size = hp.batch_size
        dataset = tf.data.TFRecordDataset(input_file_path)
        if is_training:
            dataset = dataset.shuffle(buffer_size=hp.buffer_size)
            dataset = dataset.repeat()
        if mode == "train" or mode == "dev":
            name_to_feature = {
                "in_ids": tf.FixedLenFeature([hp.max_in_length], tf.int64),
                "length": tf.FixedLenFeature([], tf.int64),
                "out_ids": tf.FixedLenFeature([hp.max_in_length], tf.int64),
                "y": tf.FixedLenFeature([hp.max_in_length], tf.int64)
            }
        elif mode == "predict":
            name_to_feature = {
                "in_ids": tf.FixedLenFeature([hp.max_in_length], tf.int64),
                "length": tf.FixedLenFeature([], tf.int64)
            }
        dataset = dataset.apply(tf.contrib.data.map_and_batch(lambda record: decoder_recoder(record, name_to_feature),
                                                              batch_size=batch_size,
                                                              drop_remainder=drop_remainder))
        return dataset
    return input_fn


if __name__ == "__main__":
    """
    a = DataLoader(hp.mode)
    a.generate_tf_file()

     """
    dataset = tf.data.TFRecordDataset(r"./data/test.recoder")
    name_to_feature = {
        "in_ids": tf.FixedLenFeature([hp.max_in_length], tf.int64),
        "length": tf.FixedLenFeature([], tf.int64)

    }
    dataset = dataset.apply(tf.contrib.data.map_and_batch(lambda record: decoder_recoder(record, name_to_feature),
                                                          batch_size=8,
                                                          drop_remainder=True))

    iter = dataset.make_initializable_iterator()
    next_value = iter.get_next()
    with tf.Session() as sess:
        sess.run(iter.initializer)
        f = sess.run(next_value)
        print(f["in_ids"].shape)
        print("\n")
        print(f["length"])
        print("\n")



