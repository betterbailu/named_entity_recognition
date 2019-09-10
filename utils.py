import pickle


def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list

def error_analysis(test_word_lists, test_tag_lists, pred_tag_lists, filepath):
    fr = open(filepath,'w')
    for i in range(len(test_tag_lists)):
        if test_tag_lists[i] != pred_tag_lists[i]:
            fr.write(str(test_word_lists[i]).replace("'",'').strip()+"\n")
            fr.write(str(test_tag_lists[i]).replace("'",'').strip()+"\n")
            fr.write(str(pred_tag_lists[i]).replace("'",'').strip()+"\n")
            for j in range(len(test_tag_lists[i])):
                if test_tag_lists[i][j] != pred_tag_lists[i][j]:
                    fr.write(str(test_word_lists[i][j])+ "\t")
                    fr.write(str(test_tag_lists[i][j])+ "\t")
                    fr.write(str(pred_tag_lists[i][j])+"\n")
            fr.write("\n")
    fr.close()

def data_clean(filename,outpath):
    out = open(outpath,"w")
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split()
        if len(line)==0:
            out.write("\n")
        else:
            line = line[0]+ "\t" + line[3]
            out.write(str(line)+"\n")
    fr.close()
    out.close()

if __name__ == "__main__":
    data_clean("./ResumeNER/eng","./ResumeNER/eng")

