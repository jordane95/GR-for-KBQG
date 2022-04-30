import json
from utils import www2fb


def load_mid2name(filename: str):
    mid2name = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) != 3:
                continue
            mid, type, name= items
            if mid not in mid2name:
                mid2name[mid] = [name]
            else:
                mid2name[mid].append(name)
    return mid2name


def convert_ids_to_names(src: str, tgt: str, mid2name):
    dataset = []

    f_src = open('src-test.txt', 'w')
    f_tgt = open('tgt-test.txt', 'w')

    with open(src, 'r') as f_in:
        qid = 0
        for line in  f_in:
            head, relation, tail, question = line.strip().split('\t')
            head_mid = www2fb(head)
            tail_mid = www2fb(tail)

            relation_name = " ".join(sum([rel.split('_') for rel in relation.split('/')[1:]], []))
            head_name = mid2name[head_mid][0] if head_mid in mid2name else 'none'
            tail_name = mid2name[tail_mid][0] if tail_mid in mid2name else 'none'
            
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

            f_src.write("\t".join([head_name, relation_name, tail_name]) + "\n")
            f_tgt.write(question + '\n')

    with open(tgt, 'w') as f_out:
        json.dump(dataset, f_out)
    
    f_src.close()
    f_tgt.close()


if __name__ == "__main__":
    mid2name = load_mid2name('names.trimmed.2M.txt')
    convert_ids_to_names('annotated_fb_data_train.txt', 'train.json', mid2name)
    convert_ids_to_names('annotated_fb_data_valid.txt', 'dev.json', mid2name)
    convert_ids_to_names('annotated_fb_data_test.txt', 'test.json', mid2name)