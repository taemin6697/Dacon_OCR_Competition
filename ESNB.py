import pandas as pd
from collections import Counter

def tolist(data):  #2번째 열만 즉, label이라고 적힌 열만 뽑아서 리스트로
  cat = data['label']
  cat_list = cat.values.tolist()
  return cat_list

def make_list(l,c, d, e,f,k):  #각 output을 동일한 label을 추론한 것끼리 리스트로 만들어줌
  com = []
  for i in range(len(c)):
    com.append([l[i],c[i], d[i], e[i],f[i],k[i]])
  return com

def most(list):  #각 단어들을 카운트해서 최빈값 만들어주는거
  most2 = []
  for i in list:
    count = Counter(i)
    most = count.most_common(n=1)
    print(most[0])
    most2.append(most[0][0])  #[('Life', 3), ('is', 3), ('too', 4), ('short', 5)]이런식으로 나와서 인덱스처리를 단어에 맞춰서 해줌
  return most2                #만약 해당 단어가 몇번 나왔나 보려면 [0]하나 지우면 됨 근데 어차피 지금은 output 2개라 의미 없긴함
data = pd.read_csv('C:/Users/tm011/PycharmProjects/pythonProject1/get_item_checkpoint-447300_cutout.csv')
data0 = pd.read_csv('C:/Users/tm011/PycharmProjects/pythonProject1/get_item_checkpoint-339948_th.csv')
data1 = pd.read_csv('C:/Users/tm011/PycharmProjects/pythonProject1/get_item_checkpoint-250488_th.csv')
data2 = pd.read_csv("C:/Users/tm011/PycharmProjects/pythonProject1/get_item_checkpoint-250488.csv")
data3 = pd.read_csv("C:/Users/tm011/PycharmProjects/pythonProject1/get_item_checkpoint-286272.csv")
data4 = pd.read_csv("C:/Users/tm011/PycharmProjects/pythonProject1/get_item_checkpoint-28627288888888.csv")
k = tolist(data)
a = tolist(data0)  #csv파일이 2개라 그냥 똑같은거로 씀
b = tolist(data1)
c = tolist(data2)
d = tolist(data3)
e = tolist(data4)

com = make_list(k,a, b, c,d,e)
last = most(com)

submit = pd.read_csv('C:/Users/tm011/PycharmProjects/pythonProject1/sample_submission.csv')
submit['label'] = last

submit.to_csv('./submission_esnb_sota_FINAL_____.csv', index=False)