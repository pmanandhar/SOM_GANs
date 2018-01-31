import minisom1
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pltt
from sklearn.datasets import load_digits as load_iris#load_digits as load_iris#load_digits as  load_iris 4#  #load_breast_cancer 30 #load_wine  13 as
from scipy import stats
from pylab import plot,axis,show,pcolor,colorbar,bone
import numpy as np
trainxx=load_iris().data
trainx=trainxx[:,[1,2,3,4]]
trainx=load_iris().data
trainy=load_iris().target
ss=20
fs=trainx.shape[1];
iterations=50000
### Initialization and training ###
som = minisom1.MiniSom1(ss,ss,fs,sigma=1.0,learning_rate=0.2)
som.random_weights_init(trainx)
print("Training...")
som.train_random(trainx,iterations) # training with 1000 iterations
print("\n...ready!")
###Extract Weight and Activation Map###
x=som.getGWeight()
y=som.getGActMap()
qnt = som.quantization(trainx)

bone()
pcolor(som.distance_map().T) # distance map as background
colorbar()
#np.mean(stats.mode(y))#np.mean(y)

markers = ['o','s','D','.','v','x','p','1','2','3','4']
colors = ['r','g','b','c','m','y','k','w','y','k','w']
inputSpace=np.zeros(qnt.shape)#np.zeros(shape=(trainx.shape[0],fs))
#winnerYIndex=np.zeros(shape=(trainx.shape[0],fs))
winnerY=np.zeros(y.shape)
for cnt,xx in enumerate(trainx):
	w = som.winner(xx) # getting the winner
	plot(w[0]+.5,w[1]+.5,markers[trainy[cnt]],markerfacecolor='None',markeredgecolor=colors[trainy[cnt]],markersize=12,markeredgewidth=2)
	ix,iy=w[0],w[1]
	inputSpace[cnt][0]=qnt[cnt][0]*y[ix][iy]
	inputSpace[cnt][1]=qnt[cnt][1]*y[ix][iy]
	inputSpace[cnt][2]=qnt[cnt][2]*y[ix][iy]
	inputSpace[cnt][3]=qnt[cnt][3]*y[ix][iy]
#show() 

print("Q.E.")
print(som.quantization_error(trainx))
print("Shape of data")
print(trainx.shape)
from scipy.spatial import distance
import numpy as np

a = inputSpace#inpSpace#np.array([(1,2,3),(1,2,3),(2,3,4),(1,1,1),(2,2,2)])
z= trainx#np.array([(1,2,3),(1,1,1)])
distance=np.zeros(a.shape[0])
for i in range(a.shape[0]):
    min=1000
    for j in range (z.shape[0]):
        dst= np.linalg.norm(a[i]-z[j])
        if (dst<=min):
            min=dst
            distance[i]=min


print("mean distance is")
print(np.mean(distance))

fig=pltt.figure()
fig.add_subplot(111,projection='3d')
pltt.plot(*trainx[:,[0,1,2]].T,linewidth=0,marker='.',markerfacecolor='k')
pltt.plot(*a[:,[0,1,2]].T,linewidth=0,marker='.',markerfacecolor='r')
pltt.show()

