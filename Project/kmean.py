import csv
import numpy as np
from collections import defaultdict

prop = [[] for _ in range(32)]


# clusster[0]: cac datatrain thuoc clusster 0
# clusster[1]: cac datatrain thuoc clusster 1
# clusster[2]: cac datatrain thuoc clusster 2
# clusster[3]: cac datatrain thuoc clusster 3
# clusster[4]: cac datatrain thuoc clusster 4

clusster = [[] for _ in range(5)]


def load_data(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [x for x in dataset[i]]
    dataset = np.array(dataset)
    return dataset



#     chuyen du lieu chu sang so
def converData(dataset):
   for k in range(len(dataset)):
    for i in range(32):
        if prop[i][0] != None:
          for j in range(len(prop[i])):
            if dataset[k][i] == prop[i][j] :
                dataset[k][i]=j
def get_data_label(dataset):
    data = []
    label = []
    for x in dataset:
        data.append(x[:31])
        label.append(x[-1])
    return data, label
# code tay
# 1. init center points
def init_centers(X):
    return X[np.random.choice(X.shape[0], 5, replace=False)]
# 2. grouping
def group_data(X, centers,weigh):
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        d=[0,0,0,0,0]
        d[0] = khoangCach(X[i], centers[0],weigh)
        d[1] = khoangCach(X[i], centers[1],weigh)
        d[2] = khoangCach(X[i], centers[2],weigh)
        d[3] = khoangCach(X[i], centers[3],weigh)
        d[4] = khoangCach(X[i], centers[4],weigh)
        y[i] = np.argmin(d)
    return y
# 3. Update center points
def update_centers(X, y, k):
    centers = np.zeros((k, X.shape[1]))
    for i in range(k):
        X_i = X[y==i, :]
        centers[i] = np.mean(X_i, axis = 0)
    return centers
# kmeans algorithm
def _kmeans(X, k,weigh):
    centers = init_centers(X)
    y = []
    while True:
        # save pre-loop groups
        y_old = y
        # grouping
        y = group_data(X, centers,weigh)
        # break while loop if groups are not changed
        if np.array_equal(y, y_old):
            break

        #  update centers
        centers = update_centers(X, y, k)
    return (centers, y)

def xacDinhCumX(X,center,weight):
    min=6;
    minKC=9999999
    for i in range(len(center)):
        if khoangCach(X,center[i],weight)<minKC:
            min=i
            minKC=khoangCach(X,center[i],weight)
    return min
def doDaiVector(X,weigh):
    count=0
    for i in range(len(X)):
        count+=X[i]*X[i]*weigh[i]
    return count
def khoangCach(X,Y,weight):
    d=X-Y;
    cout=0
    for i in range(len(d)):
        cout+=d[i]*d[i]*weight[i]
    return  cout
def xacDinhTapDataCum(dataset,center,weight):
    for i in range(len(dataset)):
        # print(i,xacDinhCumX(dataset[i],center))
        if xacDinhCumX(dataset[i],center,weight)==0:
            clusster[0].append(i)
        elif xacDinhCumX(dataset[i], center,weight) == 1:
            clusster[1].append(i)
        elif xacDinhCumX(dataset[i], center,weight) == 2:
            clusster[2].append(i)
        elif xacDinhCumX(dataset[i], center,weight) == 3:
            clusster[3].append(i)
        elif xacDinhCumX(dataset[i], center,weight) == 4:
            clusster[4].append(i)

def xacDinhNhanCum(clusster,label):
        a = [0,0,0,0,0]
        max=0
        tg=0
        for i in range(len(clusster)):
            if label[clusster[i]] == 0:
                a[0]+=1
            elif label[clusster[i]] == 1:
                a[1]+=1
            elif label[clusster[i]] == 2:
                a[2] += 1
            elif label[clusster[i]] == 3:
                a[3] += 1
            elif label[clusster[i]] == 4:
                a[4] += 1
        for i in range(5):
            a[i]=a[i]/len(clusster)

        return a

def xacDinhNhanVD(exampl,center,weight,labels):
       return labels[xacDinhCumX(exampl,center,weight)]

def xanDinhNhanCumFinal(label):
    c = [[], [], [], [], []]
    c[0]=xacDinhNhanCum(clusster[0],label)
    c[1]=xacDinhNhanCum(clusster[1], label)
    c[2]=xacDinhNhanCum(clusster[2], label)
    c[3]=xacDinhNhanCum(clusster[3], label)
    c[4]=xacDinhNhanCum(clusster[4], label)
    # xac dinh nhan cum
    clusster_label = [-1,-1,-1,-1,-1]
    a=[]
    for i in range(25):
        a.append(c[i//5][i%5])
    b=[]
    d=[]
    save=[]
    for i in range(5):
        b.append(0)
        d.append(0)
        save.append(-1)
    c = []
    for i in range(25):
        c.append(0)
    i=j=k=tg=max=0
    kt=2
    dem=25
    dem1=5
    while dem and dem1:
        max = -1.0
        for i in range(25):
            if max<a[i] and ( (a[i]<kt and a[i]>0)or(a[i]==0) ) and not c[i]:
                max= a[i]
                tg=i
        kt = max
        j = tg // 5
        k = tg % 5
        dem -= 1
        if not b[j] and not c[tg] and not d[k]:
            b[j] +=1
            c[tg]+=1
            d[k]+=1
            save[j]=k
            dem1-=1
        c[tg] = 1
    return save


def main():
    prop[0].append('GP')
    prop[0].append('MS')

    prop[1].append('F')
    prop[1].append('M' )

    prop[2].append(None)

    prop[3].append('U'       )
    prop[3].append('R'       )

    prop[4].append('LE3'     )
    prop[4].append('GT3'     )

    prop[5].append('T'       )
    prop[5].append('A'       )

    prop[6].append(None)

    prop[7].append(None)


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
    prop[12].append(None)


    prop[13].append(None)


    prop[14].append(None)


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

    prop[23].append(None)
    prop[24].append(None)
    prop[25].append(None)
    prop[26].append(None)
    prop[27].append(None)
    prop[28].append(None)
    prop[29].append(None)
    prop[30].append(None)

    prop[31].append('A')
    prop[31].append('B')
    prop[31].append('C')
    prop[31].append('D')
    prop[31].append('F')

    # trong so cua thuoc tinh
    weigh = []
    for i in range(31):

        if prop[i][0] != None:
            weigh.append(10 / len(prop[i]))
            # weigh.append(1)
        else:
            weigh.append(1)
    weigh = np.array(weigh)

    # dataset duoi dang so de dung kmean
    train = 'train1.csv'
    datasetTrain = load_data(train)
    datasetTrain = np.array(datasetTrain)
    converData(datasetTrain)
    datasetTrain = datasetTrain.astype(np.int)
    data,label=get_data_label(datasetTrain)
    data=np.array(data)
    print('Centers found by code tay:')
    centers=_kmeans(data,5,weigh)
    centers=np.array(centers[0])
    for i in range(5):
      print(centers[i][29])
    print('Xac dinh nhan cum:')
    xacDinhTapDataCum(data,centers,weigh)
    labels=xanDinhNhanCumFinal(label)
    print(labels)
    print(('Test....:'))
    datatest = load_data('test1.csv')
    test = datatest
    test = np.array(test)
    converData(test)
    test = test.astype(np.int)
    data, label = get_data_label(test)
    data = np.array(data)
    res=0
    for i in range(len(data)):
        if xacDinhNhanVD(data[i],centers,weigh,labels)==label[i]:
            res+=1
    print(res*100/len(data))

if __name__ == "__main__":
    main()




