#Chris DeGrendele
#Theory from http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
#With help on Reverse-Mode Differentiation from http://colah.github.io/posts/2015-08-Backprop/

#INITIAL DATA SET
using PyCall, Plots
sklearn_datasets = pyimport("sklearn.datasets")
X,y = sklearn_datasets.make_moons(200, noise=0.20)
X[:,1]
plot(X[:,1],X[:,2], seriestype = :scatter, color=y)
