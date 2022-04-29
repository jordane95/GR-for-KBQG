import json


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


def main(mid2name):
    f_in = open('annotated_fb_data_test.txt', 'r')
    f_src = open('src-test.txt', 'w')
    f_tgt = open('tgt-test.txt', 'w')

    for line in  f_in:
        head, relation, tail, question = line.strip().split('\t')
        head_mid = '/'.join([""] + head.split('/')[1:])
        tail_mid = '/'.join([""] + tail.split('/')[1:])
        relation_name = " ".join(relation.split('/')[-1].split('_'))
        head_name = mid2name[head_mid] if head_mid in mid2name else 'none'
        tail_name = mid2name[tail_mid] if tail_mid in mid2name else 'none'
        f_src.write("\t".join([head_name, relation_name, tail_name]) + "\n")
        f_tgt.write(question + '\n')
    f_in.close()
    f_src.close()
    f_tgt.close()


if __name__ == "__main__":
    mid2name = load_mid2name('mid2name.tsv')
    main(mid2name)