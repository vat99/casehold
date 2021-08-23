import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    full = "full"
    example = "example"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        """
        PyTorch multiple choice dataset class
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            #import pdb; pdb.set_trace()
            
            processor = processors[task]()

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                ),
            )

            cached_features_file = cached_features_file + ".pt"

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):
                #import pdb; pdb.set_trace()
                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    elif mode == Split.train:
                        examples = processor.get_train_examples(data_dir)
                    elif mode == Split.example:
                        examples = processor.get_example_examples(data_dir)
                    else:
                        examples = processor.get_all_examples(data_dir)
                    #import pdb; pdb.set_trace()
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features(
                        examples,
                        label_list,
                        max_seq_length,
                        tokenizer,
                    )
                    #import pdb; pdb.set_trace()
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def num_samples(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFMultipleChoiceDataset:
        """
        TensorFlow multiple choice dataset class
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = 128,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            logger.info(f"Creating features from dataset file at {data_dir}")
            label_list = processor.get_labels()
            if mode == Split.dev:
                examples = processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(data_dir)
            else:
                examples = processor.get_train_examples(data_dir)
            logger.info("Training examples: %s", len(examples))

            self.features = convert_examples_to_features(
                examples,
                label_list,
                max_seq_length,
                tokenizer,
            )

            def gen():
                for (ex_index, ex) in tqdm.tqdm(enumerate(self.features), desc="convert examples to features"):
                    if ex_index % 10000 == 0:
                        logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                    yield (
                        {
                            "example_id": 0,
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            self.dataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "example_id": tf.int32,
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "token_type_ids": tf.int32,
                    },
                    tf.int64,
                ),
                (
                    {
                        "example_id": tf.TensorShape([]),
                        "input_ids": tf.TensorShape([None, None]),
                        "attention_mask": tf.TensorShape([None, None]),
                        "token_type_ids": tf.TensorShape([None, None]),
                    },
                    tf.TensorShape([]),
                ),
            )

        def get_dataset(self):
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(len(self.features)))

            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class CaseHOLDProcessor(DataProcessor):
    """Processor for the CaseHOLD dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_example_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "example.csv")), "test")

    def get_all_examples(self, data_dir):
        logger.info("LOOKING AT {} all".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "all.csv")), "full")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines[1:]: # Skip line with column names
            examples.append(
                InputExample(
                    # Row index is example ID
                    example_id=line[0],
                    # No question for this task
                    question="", 
                    # Citing text prompt is context
                    contexts=[line[1], line[1], line[1], line[1], line[1]],
                    # Holding statements are endings
                    endings=[line[2], line[3], line[4], line[5], line[6]],
                    label=line[12],
                )
            )

        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    #import pdb; pdb.set_trace()
    
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    bad_count = 0
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 100 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            #logger.info("Written Features: {}".format(len(features)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )

            choices_inputs.append(inputs)

        #import pdb; pdb.set_trace()
        # try:
        #     example.label = str(int(float(example.label)))
        # except:
        #     bad_count += 1
        #     print("label example: {}, type: {}".format(example.label, type(example.label)))
        
        test_label = ""
        if example.label == "0.0":
            test_label = '0'
        elif example.label == "1.0":
            test_label = '1'
        elif example.label == "2.0":
            test_label = '2'
        elif example.label == "3.0":
            test_label = '3'
        elif example.label == "4.0":
            test_label = '4'
        
        if test_label not in label_map.keys():
            bad_count += 1
            print("bad_count: {}, label example: {}, label_map.keys: {}".format(bad_count, test_label, label_map.keys()))
            continue
        
        #print("here")
        label = label_map[test_label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        #import pdb; pdb.set_trace()
        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )
        #print("Written Features: {}".format(len(features)))

    # for f in features[:2]:
    #     logger.info("*** Example ***")
    #     logger.info("feature: %s" % f)

    return features


processors = {"casehold": CaseHOLDProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"casehold", 5}
