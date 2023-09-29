import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as ss

def majority_votes(votes):
    vote_count = {}
    for vote in votes:
        if vote in vote_count:
            vote_count[vote] += 1
        else:
            vote_count[vote] = 1
    winner = []
    for v,c in vote_count.items():
        if c == max(vote_count.values()):
            winner.append(v)
    return random.choice(winner)

def euclidian_dist(p1,p2):
    return np.sqrt(np.sum(np.power(p2-p1,2)))

def knn(points, p, k):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = euclidian_dist(p, points[i])
    ind = np.argsort(distances)
    # return (ind[:k],sorted(distances)[:k])
    return ind[:k]
            
def knn_predict(p, points, outcome, k):
    ind = knn(points,p,k)
    # print(ind)
    return majority_votes(outcome[ind])

def generate_syth_data(n = 50):
    '''
    Create two sets of points from bivariate normal distribution.
    '''
    points = np.concatenate((ss.norm(0,1).rvs((n,2)),ss.norm(1,1).rvs((n,2))),axis = 0)
    outcomes = np.concatenate((np.repeat(0,n),np.repeat(1,n)))
    return (points, outcomes)
  
def make_prediction_grid(predictors, outcomes, limits,h, k ):
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min,x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx,yy = np.meshgrid(xs,ys)
    
    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p,predictors, outcomes, k)
    
    return (xx,yy,prediction_grid)

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)
    

(predictors, outcomes) =  generate_syth_data(50)
k = 50; 
filename = 'knn_syth_5.pdf'; 
limits = (-3, 4, -3 , 4); 
h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)
plt.show()