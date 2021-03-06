niter = 63000
load_from_standardnet = True
runtraining = True
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



suffix = [
"NumberPV"
,"Ncandidates"
,"NbestTracks"
#,"mass"
,"LifeTime"
,"IPSig"
,"VertexChi2"
,"pt"
,"DOCAone"
,"DOCAtwo"
,"DOCAthree"
,"isolationa"
,"isolationb"
,"isolationc"
,"isolationd"
,"isolatione"
,"isolationf"
,"CDF3"
,"Laura_SumBDT"
,"p0_track_Chi2Dof"
,"p1_track_Chi2Dof"
,"p2_track_Chi2Dof"
,"p0_TRACKghost"
,"p0_IP"
,"p1_TRACKghost"
,"p1_IP"
,"p2_TRACKghost"
,"p2_IP"
,"SPDhits"
 ]
features = [ suff for suff in suffix ]

fmin = np.zeros(len(features))
fmax = np.zeros(len(features))
mean = np.zeros(len(features))
rms = np.zeros(len(features))

def genminmax(fnames): # all the mc files
    c = TChain("Ds2PhiPi")
    global fmin
    global fmax
    global mean
    global rms
    for fname in fnames:
        c.Add(fname)
    for i in xrange(len(features)):
        fmin[i] = c.GetMinimum(features[i])
        fmax[i] = c.GetMaximum(features[i])
        h = TH1F("buffer","buffer",400,fmin[i],fmax[i])
        c.Draw(features[i]+">>buffer")
        mean[i] = h.GetMean()
        rms[i] = h.GetRMS()
    with open("minmax.pkl","wb") as output:
        pickle.dump(fmin,output,pickle.HIGHEST_PROTOCOL)
        pickle.dump(fmax,output,pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean,output,pickle.HIGHEST_PROTOCOL)
        pickle.dump(rms, output,pickle.HIGHEST_PROTOCOL)
def getminmax():
    # http://stackoverflow.com/a/4529901/4588453
    global fmin
    global fmax
    global mean
    global rms
    with open("minmax.pkl","r") as input:
        fmin=pickle.load(input)
        fmax=pickle.load(input)
        mean=pickle.load(input)
        rms =pickle.load(input)

def convertit(fnames): # all the mc files
    c = TChain("Ds2PhiPi")
    global fmin
    global fmax
    global mean
    global rms
    for fname in fnames:
        c.Add(fname)
    train_flupshit = []
    train_labels = []
    train_jentries = []
    test_flupshit = []
    test_labels = []
    test_jentries = []
    print "reading training events"
    printed = False
    for jentry in ProgressBar()(xrange( c.GetEntries() )):
      c.GetEntry(jentry)
      #if abs(c.mctruepid)>12 :
      #    continue
      if (c.EventHash%2) == 1:
         if abs(c.truedsphipi) > 100:
           if (c.EventHash%25)<5:
            train_labels.append(1.)
            train_jentries.append(jentry)
            thelist = [getattr(c,f) for f in features]
            if not printed:
                for f in xrange(len(features)):
                    print features[f]
                    print thelist[f]
                    print fmin[f]
                    print fmax[f]
            for i in xrange(len(features)):
               if normalise:
                   thelist[i] = (thelist[i]-mean[i])/rms[i]
               else:
                   thelist[i] = (thelist[i]-fmin[i])/(fmax[i]-fmin[i]) -0.5
            if not printed:
                printed = True
                print thelist
            train_flupshit.append(thelist)
         else:
            train_labels.append(0.)
            train_jentries.append(jentry)
            thelist = [getattr(c,f) for f in features]
            for i in xrange(len(features)):
               if normalise:
                   thelist[i] = (thelist[i]-mean[i])/rms[i]
               else:
                   thelist[i] = (thelist[i]-fmin[i])/(fmax[i]-fmin[i]) -0.5
            train_flupshit.append(thelist)
      else:
         if abs(c.truedsphipi) > 100:
            test_labels.append(1.)
            test_jentries.append(jentry)
         else:
            test_labels.append(0.)
            test_jentries.append(jentry)
         thelist = [getattr(c,f) for f in features]
         for i in xrange(len(features)):
            if normalise:
                thelist[i] = (thelist[i]-mean[i])/rms[i]
            else:
                thelist[i] = (thelist[i]-fmin[i])/(fmax[i]-fmin[i]) -0.5
         test_flupshit.append(thelist)
    
    
    
    train_data = array(train_flupshit)
    train_label = array(train_labels)
    train_jentry = array(train_jentries)
    
    test_data = array(test_flupshit)
    test_label = array(test_labels)
    test_jentry = array(test_jentries)

    if grl:
       h5name = 'grltest.h5'
    else:
       h5name = 'test.h5'
    with h5py.File(h5name,'w') as f:
        f['data'] = test_data.astype(np.float32)
        f['label'] = test_label.astype(np.float32)
        f['jentry'] = test_jentry.astype(np.float32)


    if grl:
       h5name = 'grltrain.h5'
    else:
       h5name = 'train.h5'
    with h5py.File(h5name,'w') as f:
      if grl:
        f['mc_data'] = train_data.astype(np.float32)
        f['label'] = train_label.astype(np.float32)
        f['mc_jentry'] = train_jentry.astype(np.float32)
      else:
        f['data'] = train_data.astype(np.float32)
        f['label'] = train_label.astype(np.float32)
        f['jentry'] = train_jentry.astype(np.float32)
    
    if grl:


      c = TChain("Ds2PhiPi")
      c.Add("./2012UP.root")
      data_flupshit = []
      data_jentries = []
#      leftout_flupshit = []
#      leftout_jentries = []
      print "reading data"
      for jentry in ProgressBar()(xrange( c.GetEntries() )):
        c.GetEntry(jentry)
        if (c.EventHash%10) == 1:
           data_jentries.append(jentry)
           thelist = [getattr(c,f) for f in features]
           for i in xrange(len(features)):
               if normalise:
                   thelist[i] = (thelist[i]-mean[i])/rms[i]
               else:
                   thelist[i] = (thelist[i]-fmin[i])/(fmax[i]-fmin[i]) -0.5
           data_flupshit.append(thelist)
#       else:
#           leftout_jentries.append(jentry)
#           thelist = [getattr(c,f) for f in features]
#           leftout_flupshit.append(thelist)
    
      data_data = array(data_flupshit)
      data_jentry = array(data_jentries)
#      leftout_data = array(leftout_flupshit)
#      leftout_jentry = array(leftout_jentries)
    
    
      with h5py.File('realdata.h5','w') as f:
        f['real_data'] = data_data.astype(np.float32)
        f['real_jentry'] = data_jentry.astype(np.float32)
        print "real_data length   ", len(data_data)
        print "real_jentry length ", len(data_jentry)

#      with h5py.File('leftoutdata.h5','w') as f:
#        f['real_data'] = data_data.astype(np.float32)
#        f['real_jentry'] = data_jentry.astype(np.float32)



def trainit():
    global fmin
    global fmax
    global mean
    global rms
    #
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
    #import os
    ##os.chdir('/home/pseyfert/coding/caffe')
    sys.path.insert(0,'/home/pseyfert/coding/caffe/python')
    import caffe
    from pylab import *
    #%matplotlib gtk
    
    #from caffe import layers as L
    #from caffe import params as P
    #
    #def buildit(hdf5,batch_size):
    #    n = caffe.NetSpec()
    #    n.data, n.label, n.bdt, n.pid, n.sample = L.HDF5Data(bacth_size=batch_size, source = hdf5, ntop = 5)
    #    n.ip = L.InnerProduct(
    # ....
    
    
    caffe.set_mode_cpu()
    if not grl:
      solver = caffe.get_solver("solver.prototxt")
    else:
      solver = caffe.get_solver("grlsolver.prototxt")
    
    iters = np.zeros(niter+1)
    refreshers = np.zeros((niter+1)/50)
    losses = np.zeros(niter+1)
    running = np.zeros((niter+1)/50)
    testloss = np.zeros((niter+1)/50)
    dlosses = np.zeros(niter+1)
 
    if grl:
      if load_from_standardnet:
          regular = caffe.get_solver("solver.prototxt")
          regular.net.copy_from('standardsnapshot_iter_'+str(63000)+'.caffemodel')
          for i in xrange(len(solver.net.params['ip1'][0].data)):
             for j in xrange(len(solver.net.params['ip1'][0].data[0])):
                solver.net.params['ip1'][0].data[i][j] = regular.net.params['ip1'][0].data[i][j]
          for i in xrange(len(solver.net.params['ip2'][0].data)):
             for j in xrange(len(solver.net.params['ip2'][0].data[0])):
                solver.net.params['ip2'][0].data[i][j] = regular.net.params['ip2'][0].data[i][j]
          for i in xrange(len(solver.net.params['ip3'][0].data)):
             for j in xrange(len(solver.net.params['ip3'][0].data[0])):
                solver.net.params['ip3'][0].data[i][j] = regular.net.params['ip3'][0].data[i][j]

          for i in xrange(len(solver.net.params['ip1'][1].data)):
                solver.net.params['ip1'][1].data[i] = regular.net.params['ip1'][1].data[i]
          for i in xrange(len(solver.net.params['ip2'][1].data)):
                solver.net.params['ip2'][1].data[i] = regular.net.params['ip2'][1].data[i]
          for i in xrange(len(solver.net.params['ip3'][1].data)):
                solver.net.params['ip3'][1].data[i] = regular.net.params['ip3'][1].data[i]
      else:
         solver.net.copy_from('grlsnapshot_iter_12000.caffemodel')

    else:
      if load_from_standardnet:
          solver.net.copy_from('standardsnapshot_iter_'+str(39000)+'.caffemodel')
    
      
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

          #update_line(hl, (it,losses[it]))
    
        
    return losses, dlosses

def applyit():
    ##os.chdir('/home/pseyfert/coding/caffe')
    sys.path.insert(0,'/home/pseyfert/coding/caffe/python')
    import caffe
    from pylab import *
    #%matplotlib gtk
    
    #from caffe import layers as L
    #from caffe import params as P
    #
    #def buildit(hdf5,batch_size):
    #    n = caffe.NetSpec()
    #    n.data, n.label, n.bdt, n.pid, n.sample = L.HDF5Data(bacth_size=batch_size, source = hdf5, ntop = 5)
    #    n.ip = L.InnerProduct(
    # ....
    
    
    caffe.set_mode_cpu()
    if not grl:
      solver = caffe.get_solver("solver.prototxt")
    else:
      solver = caffe.get_solver("grlsolver.prototxt")
    
    losses = np.zeros(niter+1)
    dlosses = np.zeros(niter+1)
    
    if grl:
      solver.test_nets[0].copy_from('grlsnapshot_iter_'+str(niter)+'.caffemodel')
    else:
      solver.test_nets[0].copy_from('standardsnapshot_iter_'+str(niter)+'.caffemodel')
    

    #niter = 10000
    #test_interval = 100
    #train_loss = zeros(niter)
    #test_acc = zeros(int(np.ceil(niter / test_interval)))
    #output = zeros((niter,8,10))
    #for it in range(niter):
    #    solver.step(1)
    #    train_loss[it] = solver.net.blobs['loss'].data
    #    solver.test_nets[0].forward(start='fc1')
    #    output[it] = solver.test_nets[0].blobs['fc3']
    #    if it%test_interval == 0:
    #        print 'iteration', it, 'testing...'
    #        correct = 0
    
    from ROOT import TFile
    c = TChain("Ds2PhiPi")
    c.Add("./2012UP.root")
    if grl:
      of = TFile("/afs/cern.ch/work/p/pseyfert/caffe_grl.root","recreate")
    else:
      of = TFile("/afs/cern.ch/work/p/pseyfert/caffe_reg.root","recreate")
    outtree = c.CloneTree(0)
    resp = np.zeros(1,dtype=float)
    respt= np.zeros(1,dtype=float)
    if grl:
      outtree.Branch("caffe_response_grl_a",resp,"caffe_response_grl_a/D")
      outtree.Branch("caffe_response_grl_b",respt,"caffe_response_grl_b/D")
    else:
      outtree.Branch("caffe_response_a",resp,"caffe_response_a/D")
      outtree.Branch("caffe_response_b",respt,"caffe_response_b/D")
    
    #if runtraining: #REVIEW
    #  for evt in range(c.GetEntries()):
    #    solver.test_nets[0].forward()
    #    c.GetEntry(int(solver.test_nets[0].blobs['jentry'].data))
    #    resp[0] = solver.test_nets[0].blobs['lp_fc8'].data
    #    outtree.Fill()
    #else:
    for evt in ProgressBar()(range(c.GetEntries())):
        #evt = 0
        c.GetEntry(evt)
        ####if (c.EventHash%10) == 1:
        ####    continue
        thelist = [getattr(c,f) for f in features]
        for i in xrange(len(features)):
            if normalise:
                thelist[i] = (thelist[i]-mean[i])/rms[i]
            else:
                thelist[i] = (thelist[i]-fmin[i])/(fmax[i]-fmin[i]) -0.5
        solver.test_nets[0].blobs['data'].data[...] = array(thelist)
        out = solver.test_nets[0].forward(start='ip1')  #don't like out so far
        #resp[0] = solver.test_nets[0].blobs['fc5'].data
        resp[0] = solver.test_nets[0].blobs['lp_fc8'].data[0][0]
        respt[0] = solver.test_nets[0].blobs['lp_fc8'].data[0][1]
        
        outtree.Fill()
    
    of.WriteTObject(outtree)
    of.Close()

files = [
###2012UP.root
"Ds2PhiPi_2011.root",
#"Ds2PhiPi_2012.root",
"inclB.root",
"inclC.root"
]




#genminmax(files)
getminmax()

#convertit(files)

losses, dlosses = trainit()
#x = range(len(losses))
#plt.plot(x,losses)
#plt.show()
#plt.savefig('foo.png')

#applyit()
