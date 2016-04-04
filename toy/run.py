niter = 9000
grl = True
normalise = True # False = output between -.5 and .5; True = mean 0. and RMS = 1.
from ROOT import TChain, TTree, TH1F
import numpy as np
from numpy import array
import h5py
import sys
import pickle
from progressbar import ProgressBar
import matplotlib.pyplot as plt
plt.ion()
import numpy



def trainit():

    figure,ax = plt.subplots()
    lines, = ax.plot([],[])
    runninglines, = ax.plot([],[])
    testlosslines, = ax.plot([],[])
    if grl:
       dlines, = ax.plot([],[])
    ax.set_autoscaley_on(True)
    ax.set_xlim(0,niter/10)
    ax.set_ylim(0.,1.0)
    ax.grid()

    sys.path.insert(0,'/home/pseyfert/coding/caffe/python')
    import caffe
    from pylab import *
    
    caffe.set_mode_cpu()
    if not grl:
      solver = caffe.get_solver("solver.prototxt")
    else:
      solver = caffe.get_solver("grlsolver.prototxt")
      regular = caffe.get_solver("solver.prototxt")
      regular.net.copy_from('standardsnapshot_iter_'+str(9000)+'.caffemodel')
      for i in xrange(len(solver.net.params['ip1'][0].data)):
         for j in xrange(len(solver.net.params['ip1'][0].data[0])):
            solver.net.params['ip1'][0].data[i][j] = regular.net.params['ip1'][0].data[i][j]
      for i in xrange(len(solver.net.params['ip3'][0].data)):
         for j in xrange(len(solver.net.params['ip3'][0].data[0])):
            solver.net.params['ip3'][0].data[i][j] = regular.net.params['ip3'][0].data[i][j]

      for i in xrange(len(solver.net.params['ip1'][1].data)):
            solver.net.params['ip1'][1].data[i] = regular.net.params['ip1'][1].data[i]
      for i in xrange(len(solver.net.params['ip3'][1].data)):
            solver.net.params['ip3'][1].data[i] = regular.net.params['ip3'][1].data[i]


    
    iters = np.zeros(niter+1)
    refreshers = np.zeros((niter+1)/50)
    losses = np.zeros(niter+1)
    running = np.zeros((niter+1)/50)
    testloss = np.zeros((niter+1)/50)
    dlosses = np.zeros(niter+1)
 
      
    did_reset = False
    # automatic plot update taken from
    # http://stackoverflow.com/a/24272092/4588453
    for it in ProgressBar()(range( niter+1 )):
          iters[it] = it
          if (not did_reset) and it>niter/10:
              did_reset = True
              ax.set_xlim(0,niter)
          solver.step(1)
          losses[it] = solver.net.blobs['loss'].data
          if grl:
             dlosses[it] = solver.net.blobs['dc_loss'].data
             if (it%50)==10:
                 dlines.set_xdata(iters[:it])
                 dlines.set_ydata(dlosses[:it])

          if (it%50)==10:
              refreshers[it/50] = it
              running[it/50] = np.mean(losses[max(it-50,0):it])
              lines.set_xdata(iters[:it])
              lines.set_ydata(losses[:it])
              runninglines.set_xdata(refreshers[:it/50])
              runninglines.set_ydata(running[:it/50])
              tloss = []
              for i in range (30000):
                 solver.test_nets[0].forward()
                 tloss.append(solver.test_nets[0].blobs['loss'].data)
              testloss[it/50] = np.mean(tloss)
              print "TESTLOSS = ", testloss[it/50]
              testlosslines.set_xdata(refreshers[:it/50])
              testlosslines.set_ydata(testloss[:it/50])
          #if (it%1000)==10:
          #   ax.relim()
          #   ax.autoscale_view()
          if (it%50)==10:
             figure.canvas.draw()
             figure.canvas.flush_events()

    return losses, dlosses


losses, dlosses = trainit()
