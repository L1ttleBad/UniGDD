from transformers import AutoTokenizer
import json
tokenizer = AutoTokenizer.from_pretrained('/home/disk2/xcruan/doc2dial/UniGDD/output/rwth_data_t5-base')
len_list = []
for i, data in enumerate(open('/home/disk2/xcruan/doc2dial/UniGDD/task_rwth/as_sec/train.source','r').readlines()):
    seq_len = len(tokenizer.encode(data))
    if seq_len > 1000:
        print(i,seq_len)
    len_list.append(seq_len)
json.dump(len_list, open('/home/disk2/xcruan/doc2dial/UniGDD/task_rwth/as_sec/len_record.json','w'))
