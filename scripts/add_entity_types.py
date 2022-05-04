from tqdm import tqdm
import json


def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        in_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    if in_str == 'fb:m.07s9rl0':
        in_str = 'fb:m.02822'
    if in_str == 'fb:m.0bb56b6':
        in_str = 'fb:m.0dn0r'
    # Manual Correction
    if in_str == 'fb:m.01g81dw':
        in_str = 'fb:m.01g_bfh'
    if in_str == 'fb:m.0y7q89y':
        in_str = 'fb:m.0wrt1c5'
    if in_str == 'fb:m.0b0w7':
        in_str = 'fb:m.0fq0s89'
    if in_str == 'fb:m.09rmm6y':
        in_str = 'fb:m.03cnrcc'
    if in_str == 'fb:m.0crsn60':
        in_str = 'fb:m.02pnlqy'
    if in_str == 'fb:m.04t1f8y':
        in_str = 'fb:m.04t1fjr'
    if in_str == 'fb:m.027z990':
        in_str = 'fb:m.0ghdhcb'
    if in_str == 'fb:m.02xhc2v':
        in_str = 'fb:m.084sq'
    if in_str == 'fb:m.02z8b2h':
        in_str = 'fb:m.033vn1'
    if in_str == 'fb:m.0w43mcj':
        in_str = 'fb:m.0m0qffc'
    if in_str == 'fb:m.07rqy':
        in_str = 'fb:m.0py_0'
    if in_str == 'fb:m.0y9s5rm':
        in_str = 'fb:m.0ybxl2g'
    if in_str == 'fb:m.037ltr7':
        in_str = 'fb:m.0qjx99s'
    return in_str


def load_mid2name(filename: str):
    mid2name = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) != 3:
                continue
            mid, type, name= items
            if type == 'fb:type.object.name':
                mid2name[mid] = name
    return mid2name


def load_mid_to_type(path: str):
    """Load mid to type mid mapping from path

    Args:
        path (str): path to freebase subset
    Returns:
        mid2typemid (Dict[str, str]): entity mid to type entity mid mappings
    """
    mid2typemid = {}
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            head_url, relation, tail_url = line.strip().split('\t')
            head_fb_mid = www2fb(head_url)
            tail_fb_mid = www2fb(tail_url)
            if 'topic/notable_types' in relation:
                mid2typemid[head_fb_mid] = tail_fb_mid
    print(f"Loaded type entity for {len(mid2typemid)} entities.")
    return mid2typemid


def convert_ids_to_names(src: str, tgt: str, mid2name, mid2typemid):
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
            head_name = mid2name[head_mid] if head_mid in mid2name else 'none'
            tail_name = mid2name[tail_mid] if tail_mid in mid2name else 'none'

            if head_mid in mid2typemid and mid2typemid[head_mid] in mid2name:
                head_name += " ( "
                head_name += mid2name[mid2typemid[head_mid]]
                head_name += " ) "
            
            if tail_mid in mid2typemid and mid2typemid[tail_mid] in mid2name:
                tail_name += " ( "
                tail_name += mid2name[mid2typemid[tail_mid]]
                tail_name += " ) "
            
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
    mid2typemid = load_mid_to_type('freebase-subsets/freebase-FB5M.txt')
    mid2name = load_mid2name('names.trimmed.2M.txt')
    convert_ids_to_names('annotated_fb_data_train.txt', 'train.json', mid2name, mid2typemid)
    convert_ids_to_names('annotated_fb_data_valid.txt', 'dev.json', mid2name, mid2typemid)
    convert_ids_to_names('annotated_fb_data_test.txt', 'test.json', mid2name, mid2typemid)
