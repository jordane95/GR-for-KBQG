import json

TRAIN_FILE = 'annotated_fb_data_train.txt'
VALID_FILE = 'annotated_fb_data_valid.txt'
TEST_FILE = 'annotated_fb_data_test.txt'

def load_mid2name(filename: str):
    mid2name = {}
    with open(filename, 'r') as f:
        for line in f:
            datas = line.strip().split('\t')
            if len(datas) != 2:
                continue
            mid, name = datas
            mid2name[mid] = name
    return mid2name


def convert_ids_to_names(src: str, tgt: str, mid2name):
    dataset = []
    with open(src, 'r') as f_in:
        qid = 0
        for line in  f_in:
            head, relation, tail, question = line.strip().split('\t')
            head_mid = '/'.join([""] + head.split('/')[1:])
            tail_mid = '/'.join([""] + tail.split('/')[1:])
            relation_name = " ".join(relation.split('/')[-1].split('_'))
            head_name = mid2name[head_mid] if head_mid in mid2name else 'none'
            tail_name = mid2name[tail_mid] if tail_mid in mid2name else 'none'
            sample = {
                "id": qid,
                "kbs": {
                    "0": [
                        head_name,
                        head_name,
                        [
                            [
                                relation_name,
                                tail_name,
                            ]
                        ]
                    ]
                },
                "text": [
                    question,
                ]
            }
            dataset.append(sample)
            qid += 1

    with open(tgt, 'w') as f_out:
        json.dump(dataset, f_out)


if __name__ == "__main__":
    mid2name = load_mid2name('mid2name.tsv')
    convert_ids_to_names('annotated_fb_data_train.txt', 'train.json', mid2name)
    convert_ids_to_names('annotated_fb_data_valid.txt', 'dev.json', mid2name)
    convert_ids_to_names('annotated_fb_data_test.txt', 'test.json', mid2name)