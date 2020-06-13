import csv
import math
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

# x[i][j][k] = P(Xjk|ci)
x = defaultdict(lambda :defaultdict(lambda :defaultdict(int)))
#dataset have label = ci
Ci = []
# P(ci)
c = []
prop = [[] for _ in range(32)]
def load_data(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [x for x in dataset[i]]
    dataset = np.array(dataset)
    return dataset


def get_data_ci(dataset):
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    for i in range(len(dataset)):
        if dataset[i][31] == 'F':
            c0.append(dataset[i])
        elif dataset[i][31] == 'D':
            c1.append(dataset[i])
        elif dataset[i][31] == 'C':
            c2.append(dataset[i])
        elif dataset[i][31] == 'B':
            c3.append(dataset[i])
        else:
            c4.append(dataset[i])
    return c0, c1, c2, c3, c4


def sttThuoctinh(s, j):
    for i in range(len(prop[j])):
        if s == prop[j][i]:
            return i
def tinh_Pxi(dataset,xi,j,p):
    count = 0
    for i in range(len(dataset)):
          if dataset[i][j] == xi:
              count=count+1
    return (count+0.001)/(len(dataset)+0.001*p)
def tinhPci_x(y,i):
             a=0
             for j in range(31):
                 b=sttThuoctinh(y[j],j)
                 print(x[i][j][b])
                 a=a+math.log10(x[i][j][b])
             return a+math.log10(c[i])
def argMax(y):
    max=-9999999999
    tg=0
    for i in range(5):
        a=  tinhPci_x(y,i)
        if max<a:
            max=a
            tg=i
            print(a)
    return tg
def getLable(x):
    if x=='A':
        return 4
    elif x=='B':
        return 3
    elif x=='C':
        return 2
    elif x=='D':
        return 1
    elif x=='F':
        return 0

#     chuyen du lieu chu sang so
def converData(dataset):
   for k in range(len(dataset)):
    for i in range(32):
        for j in range(len(prop[i])):
            if dataset[k][i] == prop[i][j]:
                dataset[k][i]=j
def main():
    train = 'train.csv'
    datasetTrain = load_data(train)
    a,b,f,d,e = get_data_ci(datasetTrain)
    Ci.append(np.array(a))
    Ci.append(np.array(b))
    Ci.append(np.array(f))
    Ci.append(np.array(d))
    Ci.append(np.array(e))
    for i in range(len(Ci)):
        c.append(len(Ci[i]) / len(datasetTrain)  )
    prop[0].append('GP')
    prop[0].append('MS')
    prop[1].append('F')
    prop[1].append('M' )
    prop[2].append('15'      )
    prop[2].append('16'      )
    prop[2].append('17'      )
    prop[2].append('18'      )
    prop[2].append('19'       )
    prop[2].append('20'      )
    prop[2].append('21'      )
    prop[2].append('22'      )

    prop[3].append('U'       )
    prop[3].append('R'       )

    prop[4].append('LE3'     )
    prop[4].append('GT3'     )

    prop[5].append('T'       )
    prop[5].append('A'       )

    prop[6].append('0'       )
    prop[6].append('1'       )
    prop[6].append('2'       )
    prop[6].append('3'       )
    prop[6].append('4'       )

    prop[7].append('0'       )
    prop[7].append('1'       )
    prop[7].append('2'       )
    prop[7].append('3'       )
    prop[7].append('4'       )

    prop[8].append('teacher' )
    prop[8].append('health'   )
    prop[8].append('services')
    prop[8].append('at_home' )
    prop[8].append('other'   )

    prop[9].append('teacher' )
    prop[9].append('health'   )
    prop[9].append('services')
    prop[9].append('at_home' )
    prop[9].append('other'   )

    prop[10].append('home'      )
    prop[10].append('reputation')
    prop[10].append('course'    )
    prop[10].append('other'     )

    prop[11].append('mother'    )
    prop[11].append('father'    )
    prop[11].append('other')
    prop[12].append('1'    )
    prop[12].append('2'    )
    prop[12].append('3'    )
    prop[12].append('4'    )

    prop[13].append('1'    )
    prop[13].append('2'    )
    prop[13].append('3'    )
    prop[13].append('4'    )

    prop[14].append('0'    )
    prop[14].append('1'    )
    prop[14].append('2'    )
    prop[14].append('3'    )

    prop[15].append('yes'  )
    prop[15].append('no'   )

    prop[16].append('yes'  )
    prop[16].append('no'   )

    prop[17].append('yes'  )
    prop[17].append('no'   )

    prop[18].append('yes'  )
    prop[18].append('no'   )

    prop[19].append('yes'  )
    prop[19].append('no'   )

    prop[20].append('yes'  )
    prop[20].append('no'   )

    prop[21].append('yes'  )
    prop[21].append('no'   )

    prop[22].append('yes'  )
    prop[22].append('no'   )

    prop[23].append('1'    )
    prop[23].append('2'    )
    prop[23].append('3'    )
    prop[23].append('4'    )
    prop[23].append('5'    )

    prop[24].append('1'    )
    prop[24].append('2'    )
    prop[24].append('3'    )
    prop[24].append('4'    )
    prop[24].append('5'    )

    prop[25].append('1'    )
    prop[25].append('2'    )
    prop[25].append('3'    )
    prop[25].append('4'    )
    prop[25].append('5'    )

    prop[26].append('1'    )
    prop[26].append('2'    )
    prop[26].append('3'    )
    prop[26].append('4'    )
    prop[26].append('5'    )

    prop[27].append('1'    )
    prop[27].append('2'    )
    prop[27].append('3'    )
    prop[27].append('4'    )
    prop[27].append('5'    )

    prop[28].append('1'    )
    prop[28].append('2'    )
    prop[28].append('3'    )
    prop[28].append('4'    )
    prop[28].append('5'    )

    prop[29].append('A'    )
    prop[29].append('B'    )
    prop[29].append('C'    )
    prop[29].append('D'    )
    prop[29].append('F'    )

    prop[30].append('A'    )
    prop[30].append('B'    )
    prop[30].append('C'    )
    prop[30].append('D'    )
    prop[30].append('F'    )

    prop[31].append('A')
    prop[31].append('B')
    prop[31].append('C')
    prop[31].append('D')
    prop[31].append('F')
    for i in range (5):
        for j in range(31):
            a=0
            for k in range(len(prop[j])):
                 x[i][j][k]=tinh_Pxi(Ci[i],prop[j][k],j,len(prop[j]))

    datatest = load_data('test1.csv')
    res=0
    count=len(datatest)
    for i in range(count):
          if argMax(datatest[i])==getLable(datatest[i][31]):
              res+=1
    print('arc:',res*100/count)
    print("-------------------------")


if __name__ == "__main__":
    main()




