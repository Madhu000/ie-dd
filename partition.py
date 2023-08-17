from sentence_transformers import SentenceTransformer, util
import os
import csv
import time
import pickle as pkl
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import glob


def get_nw_label(data,b_tag,i_tag):
  new_label=[]
  for j in data:
    if j.startswith('B-'):
      nw_lbl=b_tag
      new_label.append(nw_lbl)
    elif j.startswith('I-'):
      nw_lbl=i_tag
      new_label.append(nw_lbl)
    else:
      new_label.append('O')
  return new_label

def get_partitioned_cluster(data, dir_path_tr, dir_path_val, flname, b_tag, i_tag):
  dir_path_train=dir_path_tr
  dir_path_valid=dir_path_val
  fl_path_tr=os.path.join(dir_path_train,flname)
  fl_path_val=os.path.join(dir_path_valid,flname)
  if not os.path.exists(dir_path_train):
    os.makedirs(dir_path_train)
  if not os.path.exists(dir_path_valid):
    os.makedirs(dir_path_valid)
  fl_tr_w=open(fl_path_tr,"wb")
  fl_val_w=open(fl_path_val,"wb")
  train_data_nw=OrderedDict()
  valid_data_nw=OrderedDict()
  for i in tqdm(data):
    if i in data_train_2:
      tokens_tr=data_train_2[i][0]
      labels_tr=data_train_2[i][-1]
      labels_nw_tr=get_nw_label(labels_tr,b_tag,i_tag)
      nw_value_tr=[]
      nw_value_tr.append(tokens_tr)
      nw_value_tr.append(labels_nw_tr)
      train_data_nw[i]=nw_value_tr
    elif i in data_valid_2:
      tokens_vl=data_valid_2[i][0]
      labels_vl=data_valid_2[i][-1]
      labels_nw_vl=get_nw_label(labels_vl,b_tag,i_tag)
      nw_value_vl=[]
      nw_value_vl.append(tokens_vl)
      nw_value_vl.append(labels_nw_vl)
      valid_data_nw[i]=nw_value_vl
  pkl.dump(train_data_nw, fl_tr_w)
  pkl.dump(valid_data_nw, fl_val_w)
  fl_tr_w.close()
  fl_val_w.close()
  return 0

def get_txt_file(fl_w,fl_paths):
  fl_trn=open(fl_w,"w")
  for i in tqdm(fl_paths):
    data=pkl.load(open(i,"rb"))
    #sum_=sum_+len(data)
    for j in data:
      tkns=data[j][0]
      lbls=data[j][-1]
      if len(tkns)>65:
        #count_=count_+1
        continue
      else:
        for k,l in enumerate(tkns):
          fl_trn.write(l)
          fl_trn.write(" ")
          fl_trn.write(lbls[k])
          fl_trn.write("\n")
        fl_trn.write("\n")
        #count=count+1
  fl_trn.close()
  return 0

def get_required_dict(data):
  global_dict_train_1=OrderedDict()
  global_dict_train_2=OrderedDict()
  tmp_ls=[]
  tmp_label=[]
  for i in tqdm(data):
    tmp=i.strip('\n').split(' ')
    if len(tmp)<=1:
      txt=" ".join(tmp_ls)
      if txt in global_dict_train_2:
        global_dict_train_1[txt]=[]
        global_dict_train_1[txt].append(tmp_ls)
        global_dict_train_1[txt].append(tmp_label)
      else:
        global_dict_train_2[txt]=[]
        global_dict_train_2[txt].append(tmp_ls)
        global_dict_train_2[txt].append(tmp_label)
      tmp_ls=[]
      tmp_label=[]
    else:
      wrd=tmp[0]
      label=tmp[-1]
      tmp_ls.append(wrd)
      tmp_label.append(label)
  return global_dict_train_1, global_dict_train_2

fl_train=open("data/Finegrained_Flair_data/train_nw.txt","r")
fl_data_train=fl_train.readlines()
_,data_train_2=get_required_dict(fl_data_train)

fl_valid=open("data/Finegrained_Flair_data/valid_nw.txt","r")
fl_data_valid=fl_valid.readlines()
_,data_valid_2=get_required_dict(fl_data_valid)

fl_test=open("data/Finegrained_Flair_data/test_nw.txt","r")
fl_data_test=fl_test.readlines()
_,data_test_2=get_required_dict(fl_data_test)

corpus_sentences = list(data_train_2.keys())+list(data_valid_2.keys())

corpus_embeddings=pkl.load(open('data/nw_cluster/corpus_embed/corpus_embed.pkl','rb'))

kmeans = KMeans(n_clusters=7)
tqdm(kmeans.fit(corpus_embeddings))

for i in range(1,8):
  name="cluster_{}".format(i)
  globals()[name]=[]

global_lst=[]
for i in range(corpus_embeddings.shape[0]):
  if kmeans.labels_[i] == 0:
    cluster_1.append(corpus_sentences[i])
  elif kmeans.labels_[i] == 1:
    cluster_2.append(corpus_sentences[i])
  elif kmeans.labels_[i]==2:
    cluster_3.append(corpus_sentences[i])
  elif kmeans.labels_[i]==3:
    cluster_4.append(corpus_sentences[i])
  elif kmeans.labels_[i]==4:
    cluster_5.append(corpus_sentences[i])
  elif kmeans.labels_[i]==5:
    cluster_6.append(corpus_sentences[i])
  elif kmeans.labels_[i]==6:
    cluster_7.append(corpus_sentences[i])
global_lst.append(cluster_1)
global_lst.append(cluster_2)
global_lst.append(cluster_3)
global_lst.append(cluster_4)
global_lst.append(cluster_5)
global_lst.append(cluster_6)
global_lst.append(cluster_7)

a=get_partitioned_cluster(cluster_1, "data/nw_cluster/kmeans/seven/train_data" ,"data/nw_cluster/kmeans/seven/val_data", "cluster_1.pkl", "B-ONE","I-ONE")

b=get_partitioned_cluster(cluster_2, "data/nw_cluster/kmeans/seven/train_data" ,"data/nw_cluster/kmeans/seven/val_data", "cluster_2.pkl", "B-TWO","I-TWO")

c=get_partitioned_cluster(cluster_3, "data/nw_cluster/kmeans/seven/train_data" ,"data/nw_cluster/kmeans/seven/val_data", "cluster_3.pkl", "B-THRE","I-THRE")

d=get_partitioned_cluster(cluster_4, "data/nw_cluster/kmeans/seven/train_data" ,"data/nw_cluster/kmeans/seven/val_data", "cluster_4.pkl", "B-FOR","I-FOR")

e=get_partitioned_cluster(cluster_5, "data/nw_cluster/kmeans/seven/train_data" ,"data/nw_cluster/kmeans/seven/val_data", "cluster_5.pkl", "B-FVE","I-FVE")

f=get_partitioned_cluster(cluster_6, "data/nw_cluster/kmeans/seven/train_data" ,"data/nw_cluster/kmeans/seven/val_data", "cluster_6.pkl", "B-SIX","I-SIX")

g=get_partitioned_cluster(cluster_7, "data/nw_cluster/kmeans/seven/train_data" ,"data/nw_cluster/kmeans/seven/val_data", "cluster_7.pkl", "B-SVN","I-SVN")

"""##Testing"""

corpus_sentences_tst=list(data_test_2.keys())

corpus_sentences_tst=pkl.load(open("data/nw_cluster/corpus_embed/corpus_embed_tst.pkl","rb"))

tst_lst=kmeans.predict(corpus_sentences_tst)

for i in range(1,8):
  name="cluster_{}".format(i)
  globals()[name]=[]
  #print(cluster_1)

global_lst=[]
for i in range(np_arr_tst.shape[0]):
  if tst_lst[i] == 0:
    cluster_1.append(corpus_sentences_tst[i])
  elif tst_lst[i] == 1:
    cluster_2.append(corpus_sentences_tst[i])
  elif tst_lst[i]==2:
    cluster_3.append(corpus_sentences_tst[i])
  elif tst_lst[i]==3:
    cluster_4.append(corpus_sentences_tst[i])
  elif tst_lst[i]==4:
    cluster_5.append(corpus_sentences_tst[i])
  elif tst_lst[i]==5:
    cluster_6.append(corpus_sentences_tst[i])
  elif tst_lst[i]==6:
    cluster_7.append(corpus_sentences_tst[i])
  # elif tst_lst[i]==7:
  #   cluster_8.append(corpus_sentences_tst[i])
global_lst.append(cluster_1)
global_lst.append(cluster_2)
global_lst.append(cluster_3)
global_lst.append(cluster_4)
global_lst.append(cluster_5)
global_lst.append(cluster_6)
global_lst.append(cluster_7)

def get_partitioned_cluster_tst(data, dir_path_tst, flname, b_tag, i_tag):
  # dir_path_train=dir_path_tr
  # dir_path_valid=dir_path_val
  count=0
  dir_path_tst_=dir_path_tst
  fl_path_test=os.path.join(dir_path_tst_,flname)
  if not os.path.exists(dir_path_tst_):
    os.makedirs(dir_path_tst_)
  fl_tst_w=open(fl_path_test,"wb")
  test_data_nw=OrderedDict()
  for i in tqdm(data):
    if i in data_test_2:
      tokens_tr=data_test_2[i][0]
      labels_tr=data_test_2[i][-1]
      labels_nw_tr=get_nw_label(labels_tr,b_tag,i_tag)
      nw_value_tr=[]
      nw_value_tr.append(tokens_tr)
      nw_value_tr.append(labels_nw_tr)
      test_data_nw[i]=nw_value_tr
      count=count+1
  pkl.dump(test_data_nw, fl_tst_w)
  fl_tst_w.close()
  print(count)
  return 0

a=get_partitioned_cluster_tst(cluster_1, "data/nw_cluster/kmeans/seven/test_data" , "cluster_1.pkl", "B-ONE","I-ONE")

b=get_partitioned_cluster_tst(cluster_2, "data/nw_cluster/kmeans/seven/test_data" , "cluster_2.pkl", "B-TWO","I-TWO")

c=get_partitioned_cluster_tst(cluster_3, "data/nw_cluster/kmeans/seven/test_data" , "cluster_3.pkl", "B-THRE","I-THRE")

d=get_partitioned_cluster_tst(cluster_4, "data/nw_cluster/kmeans/seven/test_data", "cluster_4.pkl", "B-FOR","I-FOR")

e=get_partitioned_cluster_tst(cluster_5, "data/nw_cluster/kmeans/seven/test_data" , "cluster_5.pkl", "B-FVE","I-FVE")

f=get_partitioned_cluster_tst(cluster_6, "data/nw_cluster/kmeans/seven/test_data" , "cluster_6.pkl", "B-SIX","I-SIX")

g=get_partitioned_cluster_tst(cluster_7, "data/nw_cluster/kmeans/seven/test_data" , "cluster_7.pkl", "B-SVN","I-SVN")

"""##Training and Test data of txt file preparation"""

fl_paths=glob.glob("data/nw_cluster/kmeans/seven/train_data/*.pkl")

fl_paths_val=glob.glob("data/nw_cluster/kmeans/seven/val_data/*.pkl")

fl_paths_test=glob.glob("data/nw_cluster/kmeans/seven/test_data/*.pkl")

fl_w="data/nw_cluster/kmeans/FG/7_cluster_768_scibert/train_nw.txt"
a=get_txt_file(fl_w,fl_paths)

fl_w="data/nw_cluster/kmeans/FG/7_cluster_768_scibert/valid_nw.txt"
b=get_txt_file(fl_w,fl_paths_val)

fl_w="data/nw_cluster/kmeans/FG/7_cluster_768_scibert/test_nw.txt"
b=get_txt_file(fl_w,fl_paths_test)