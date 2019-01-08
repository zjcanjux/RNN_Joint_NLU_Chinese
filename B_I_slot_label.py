# -*- coding: utf-8 -*-

# 中文语料B_I槽标注
import jieba

input_file_path = "dataset/chinese_data/seq_slot_intent.text"
output_file_path = 'dataset/chinese_data/processed_data_2.text'

class Slot_label(object):
    """docstring for Slot_label"""
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def B_I_label(self):
        f = open(self.input_file_path, 'r').readlines()
        seq_cut_total = []
        slot_name_cut_total = []
        slot_tag_total = []
        intent_total = []

        for i in f:
            seq, slot_name, slot_tag, intent = i.split()
            seq_cut = jieba.lcut(seq)
            slot_name = jieba.lcut(slot_name)
            b = []
            for i in seq_cut:
                if b == []:
                    b.append('BOS')
                    b.append(' ')
                    b.append(i)
                    b.append(' ')
                else:
                    b.append(i)
                    b.append(' ')
            b.append('EOS')
            seq_cut_total.append(b)
            slot_name_cut_total.append(slot_name)
            slot_tag_total.append(slot_tag)
            intent_total.append(intent)
            e = [[m,n,o,p] for m,n,o,p in zip(seq_cut_total, slot_name_cut_total, slot_tag_total, intent_total)]

        new_slot_tag_total= []
        for i in e:
            test_seq, test_slot_name, test_slot_tag, test_intent = i
            new_slot_tag = []
            y = 0
            num_in_test_slot_name = 0
            # 判断句子是是否有槽
            for i in test_seq:
                if i in test_slot_name:
                    num_in_test_slot_name +=1
            # 句子中的实体和槽名不一致的情况
            money_num = ['一元','二元','三元','三块','五元','五块','六元','六块钱','九元','九块','十元','十块','十块钱','十五元',
                            '二十元','二十块','三十元','三十块','三十','四十元','四十块','八元',
                            '五十元','五十块','有九元','三十八', '两百多','两百块', '一百二十块',
                            '六十元','六十块','七十元','七十','七十块','九十元','九十块','一百元','一百块','一百','一百二十五','一百二十五元',
                            '一百五十元','一百五','一百五十','一百五十块','六十六', '二百元','二百','二百块','三百元','三百','三百块','一个七十']
            month_num = ['一月份','二月份','三月份','四月份','五月份','六月份','七月份','八月份','九月份','十月份','十一月份','十二月份',
                        '月','本月','上个月','月话费','这月花','当月','七月','一个月', '九月','八月','六月','下个月','月底','五月']
            key_token = ['密钥','秘钥','秘密','新密码']
            day_turnover = ['一天']
            # 若句子没有槽，则判断是不是有数字的情况
            if num_in_test_slot_name == 0:
                for i in test_seq:
                    if i in money_num:
                        i = 'B-'+'金额'
                        new_slot_tag.append(i)
                        continue
                    if i in month_num:
                        i = 'B-'+'月份'
                        new_slot_tag.append(i)
                        continue
                    if i in key_token:
                        i = 'B-'+'附属标签'
                        new_slot_tag.append(i)
                        continue
                    if i in day_turnover:
                        i = 'B-流量业务对象类型'
                        new_slot_tag.append(i)
                        continue
                    if i == '九十八' or i == '九十八块' or i == '十八' or i == '一百七十' or i == '五十八':
                        i = 'B-'+'数字'
                        new_slot_tag.append(i)
                        continue
                    if i in test_slot_name and y == 0:
                        i = 'B-'+test_slot_tag
                        new_slot_tag.append(i)
                        y +=1
                    elif i in test_slot_name and y>0:
                        i = 'I-'+test_slot_tag
                        new_slot_tag.append(i)
                    elif i == ' ':
                        new_slot_tag.append(' ')
                    else:
                        new_slot_tag.append('O')
            # 若句子中有槽，则按预先设定的槽进行标注
            if num_in_test_slot_name > 0:
                for i in test_seq:

                    if i in test_slot_name and y == 0:
                        i = 'B-'+test_slot_tag
                        new_slot_tag.append(i)
                        y +=1
                    elif i in test_slot_name and y>0:
                        i = 'I-'+test_slot_tag
                        new_slot_tag.append(i)
                    elif i == ' ':
                        new_slot_tag.append(' ')
                    else:
                        new_slot_tag.append('O')
            new_slot_tag_total.append(new_slot_tag)

        g = [[m,n,o,p,q] for m,n,o,p,q in zip(seq_cut_total, slot_name_cut_total, new_slot_tag_total, slot_tag_total,intent_total)]
        # 生成输出文件
        fw = open(self.output_file_path,'w')
        for i in g:
            for j in i[0]:
                fw.write(j)
            fw.write('\t')
            for j in i[2]:
                fw.write(j)
            fw.write(' ')
            for j in i[4]:
                fw.write(j)
            fw.write('\n')


aa = Slot_label(input_file_path, output_file_path)
aa.B_I_label()
