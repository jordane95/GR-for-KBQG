
f_in = open('annotated_fb_data_test.txt', 'r')
f_src = open('src-test.txt', 'w')
f_tgt = open('tgt-test.txt', 'w')

for line in f_in:
    head, relation, tail, question = line.strip().split('\t')
    f_src.write(" ".join([head, relation, tail]) + "\n")
    f_tgt.write(question + "\n")

f_in.close()
f_src.close()
f_tgt.close()