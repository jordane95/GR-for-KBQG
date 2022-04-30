import os
import math

import pandas as pd
import numpy as np


class Data:
    """
    class to read from zero shot processed data and convert to text

    """

    def __init__(self, datapath=None, seed=3, remove_unk=False):

        self.remove_unk = remove_unk

        if datapath is None:
            datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./preprocessed")

        np.random.seed(seed)

        # loading vocab
        def return_vocab(filename):

            keys = [l.strip() for l in open(filename).readlines()]
            values = range(0, len(keys))
            return dict(zip(keys, values))

        def return_inv_vocab(filename):

            keys = [l.strip() for l in open(filename).readlines()]
            values = range(0, len(keys))
            return dict(zip(values, keys))

        self.entityvocab = return_vocab(os.path.join(datapath, "entity.vocab"))
        self.propertyvocab = return_vocab(os.path.join(datapath, "property.vocab"))
        self.wordvocab = return_vocab(os.path.join(datapath, "word.vocab"))

        self.inv_entityvocab = return_inv_vocab(os.path.join(datapath, "entity.vocab"))
        self.inv_propertyvocab = return_inv_vocab(os.path.join(datapath, "property.vocab"))
        self.inv_wordvocab = return_inv_vocab(os.path.join(datapath, "word.vocab"))

        # loading data files names
        self.datafile = {"train":os.path.join(datapath, "train.ids"),
                         "valid": os.path.join(datapath, "valid.ids"),
                         "test": os.path.join(datapath, "valid.ids")
                         }

        self.data = {}

    def read_data(self, mode):
        # modes = ["train", "test", "valid"]
        data = []

        f = self.datafile[mode]
        x = pd.read_csv(f, names=["sub", "pred", "obj", "question", "subtype", "objtype", "dep", "direction", "placeholder_dict"])

        placeholder_dicts = [eval(i) for i in x['placeholder_dict'].values] # List[List[Tuple[str, str]]]
        subjects = [
            subject
            for placeholder_dict in placeholder_dicts
            for subject, placeholder in placeholder_dict
            if placeholder == '_PLACEHOLDER_SUB_'
        ]

        objects = [
            obj
            for placeholder_dict in placeholder_dicts
            for obj, placeholder in placeholder_dict
            if placeholder == '_PLACEHOLDER_SUB_'
        ]





        if self.remove_unk:
            unkdep = self.wordvocab["_UNK_DEP_"] if "_UNK_DEP_" in self.wordvocab else None
            x = x[x.dep != unkdep]
            x = x[x.apply(lambda i: str(self.wordvocab["_PLACEHOLDER_SUB_"]) in i['question'].split(), axis=1)]

        x.reset_index(inplace=True)

        tmp = [[], [], [], []]
        for l in x.iterrows():

            tmp[0].append([int(i) for i in l[1]['question'].split()])
            tmp[1].append([int(i) for i in l[1]['subtype'].split()])
            tmp[2].append([int(i) for i in l[1]['objtype'].split()])
            tmp[3].append([int(i) for i in l[1]['dep'].split()])

        x['question'] = tmp[0]
        x['subtype'] = tmp[1]
        x['objtype'] = tmp[2]
        x['dep'] = tmp[3]

        x['question_length'] = x.apply(lambda l: len(l['question']), axis=1)
        x['subtype_length'] = x.apply(lambda l: len(l['subtype']), axis=1)
        x['objtype_length'] = x.apply(lambda l: len(l['objtype']), axis=1)
        x['dep_length'] = x.apply(lambda l: len(l['dep']), axis=1)
        x['triple'] = x.apply(lambda l: [l['sub'], l['pred'], l['obj']], axis=1)

        return x

    def datafeed(self, mode, config, shuffle=True):
        """

        :param mode: train, valid, test
        :param config: config object
        :param shot_percentage: float between 0 and 1 indicating the percentage of the training data taken into consideration
        :param min_count: int indicating the minimum count of the predicates of the examples being taken in to consideration
        :param shuffle: whether to shuffle the training data or not
        :param kfold: a number between 1 and 10
        :return:
        """

        x = self.read_data(mode)
        self.data[mode] = x

        dataids = x.index

        if shuffle:
            np.random.shuffle(dataids)

        return self.yield_datafeed(mode, dataids, x, config)


    def yield_datafeed(self, mode, dataids, x, config):
        """
        given a dataframe and selected ids and a mode yield data for experiments
        :param mode:
        :param dataids:
        :param x:
        :param config:
        :return:
        """

        if mode == "train":

            for epoch in range(config.MAX_EPOCHS):

                def chunks(l, n):
                    """Yield successive n-sized chunks from l."""
                    for i in range(0, len(l), n):
                        yield l[i:i + n]

                for bn, batchids in enumerate(chunks(dataids, config.BATCH_SIZE)):

                    batch = x.iloc[batchids]

                    max_length = max([batch['subtype_length'].values.max(),
                         batch['objtype_length'].values.max(),
                         batch['dep_length'].values.max(),
                         ])

                    yield (
                        np.array([i for i in batch['triple'].values]),
                        self.pad(batch['subtype'].values, max_length=max_length),
                        batch['subtype_length'].values,
                        self.pad(batch['objtype'].values, max_length=max_length),
                        batch['objtype_length'].values,
                        self.pad(batch['dep'].values, max_length=max_length),
                        batch['dep_length'].values,
                        self.pad(batch['question'].values),
                        batch['question_length'].values,
                        batch['direction'].values,
                        {"epoch": epoch, "batch_id": bn, "ids": batchids, "placeholder_dict":[eval(i) for i in batch["placeholder_dict"].values]}  # meta info
                    )

        if mode == "test" or mode == "valid":
            # in case of test of validation batch of size 1 and no shuffle
            # takes longer computation time but allows variable lengths

            for id in dataids:

                batch = x.iloc[[id]]

                max_length = max([batch['subtype_length'].values.max(),
                                  batch['objtype_length'].values.max(),
                                  batch['dep_length'].values.max(),
                                  ])

                yield (
                    np.array([i for i in batch['triple']]),
                    self.pad(batch['subtype'], max_length=max_length),
                    batch['subtype_length'].values,
                    self.pad(batch['objtype'], max_length=max_length),
                    batch['objtype_length'].values,
                    self.pad(batch['dep'].values, max_length=max_length),
                    batch['dep_length'].values,
                    self.pad(batch['question'].values),
                    batch['question_length'].values,
                    batch['direction'].values,
                    {"ids": id, "placeholder_dict": [eval(i) for i in batch["placeholder_dict"].values]}  # meta info
                )

    def pad(self, x, pad_char=0, max_length=None):
        """
        helper function to add padding to a batch
        :param x: array of arrays of variable length
        :return:  x with padding of max length
        """

        if max_length is None:
            max_length = max([len(i) for i in x])

        y = np.ones([len(x), max_length]) * pad_char

        for c, i in enumerate(x):
            y[c, :len(i)] = i

        return y
