niter = 39000
runtraining = True
grl = True
from ROOT import TChain, TTree
import numpy as np
from numpy import array
import h5py
import sys
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import numpy



suffix = [
"NumberPV"
,"Ncandidates"
,"NbestTracks"
,"mass"
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

def update_line(hl, new_data):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_data[0]))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data[1]))
    plt.draw()

fmin = np.zeros(len(features))
fmax = np.zeros(len(features))

def genminmax(fnames): # all the mc files
    c = TChain("Ds2PhiPi")
    global fmin
    global fmax
    for fname in fnames:
        c.Add(fname)
    for i in xrange(len(features)):
        fmin[i] = c.GetMinimum(features[i])
        fmax[i] = c.GetMaximum(features[i])
def convertit(fnames): # all the mc files
    c = TChain("Ds2PhiPi")
    global fmin
    global fmax
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
            thelist[i] = (thelist[i]-fmin[i])/(fmax[i]-fmin[i]) -0.5
         test_flupshit.append(thelist)
    
    
    
    train_data = array(train_flupshit)
    train_label = array(train_labels)
    train_jentry = array(train_jentries)
    
    test_data = array(test_flupshit)
    test_label = array(test_labels)
    test_jentry = array(test_jentries)
    
    with h5py.File('test.h5','w') as f:
        f['data'] = test_data.astype(np.float32)
        f['label'] = test_label.astype(np.float32)
        f['jentry'] = test_jentry.astype(np.float32)
    
    
    with h5py.File('train.h5','w') as f:
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

#      with h5py.File('leftoutdata.h5','w') as f:
#        f['real_data'] = data_data.astype(np.float32)
#        f['real_jentry'] = data_jentry.astype(np.float32)



def trainit(hl):
    global fmin
    global fmax
    #
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
    
    losses = np.zeros(niter+1)
    dlosses = np.zeros(niter+1)
    
      
    for it in ProgressBar()(range( niter+1 )):
          solver.step(1)
          losses[it] = solver.net.blobs['loss'].data
          if grl:
             dlosses[it] = solver.net.blobs['dc_loss'].data

          ### fixme
          #update_line(hl, (it,losses[it]))
    
        
    resps = np.zeros((100,2))
    trues = np.zeros(100)
    for i in range (100):
        solver.test_nets[0].forward()
        resps[i] = solver.test_nets[0].blobs['lp_fc8'].data
        trues[i] = solver.test_nets[0].blobs['label'].data
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
      of = TFile("caffe_grl.root","recreate")
    else:
      of = TFile("caffe_reg.root","recreate")
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
genminmax(files)
convertit(files)

hl, = plt.plot([], [])

losses, dlosses = trainit(hl)

x = range(len(losses))
plt.plot(x,losses)
plt.show()
plt.savefig('foo.png')

#applyit()



###if add_to_ks:
###    kschain = TChain("Tuple")
###    kschain.Add("ks.root")
###    respp = np.zeros(1,dtype=float)
###    respm = np.zeros(1,dtype=float)
###    resptp= np.zeros(1,dtype=float)
###    resptm= np.zeros(1,dtype=float)
###
###    if grl:
###       of = TFile("ks_friend_grl.root","recreate")
###       outtree = TTree("ks_friend_grl","ks friend with grl")
###       outtree.Branch("piplus_caffe_responsec",respp,"caffe_responsec/D")
###       outtree.Branch("piplus_caffe_responsed",resptp,"caffe_responsed/D")
###       outtree.Branch("piminus_caffe_responsec",respm,"caffe_responsec/D")
###       outtree.Branch("piminus_caffe_responsed",resptm,"caffe_responsed/D")
###    else:
###       of = TFile("ks_friend.root","recreate")
###       outtree = TTree("ks_friend","ks friend without grl")
###       outtree.Branch("piplus_caffe_responsea",respp,"caffe_responsea/D")
###       outtree.Branch("piplus_caffe_responseb",resptp,"caffe_responseb/D")
###       outtree.Branch("piminus_caffe_responsea",respm,"caffe_responsea/D")
###       outtree.Branch("piminus_caffe_responseb",resptm,"caffe_responseb/D")
###    for evt in ProgressBar()(xrange(kschain.GetEntries())):
###        kschain.GetEntry(evt)
###        thelist = [getattr(kschain,f) for f in piplusfeatures]
###        thelist.append(1.)
###        for ivar in range(21):
###           thelist[ivar] = thelist[ivar]-fMin_1[2][ivar]
###           thelist[ivar] = thelist[ivar]*fscale[2][ivar] - 1.
###        solver.test_nets[0].blobs['data'].data[...] = array(thelist)
###        out = solver.test_nets[0].forward(start='ip1')  #don't like out so far
###        #resp[0] = solver.test_nets[0].blobs['fc5'].data
###        respp[0] = solver.test_nets[0].blobs['lp_fc8'].data[0][0]
###        resptp[0] = solver.test_nets[0].blobs['lp_fc8'].data[0][1]
###        thelist = [getattr(kschain,f) for f in piminusfeatures]
###        thelist.append(1.)
###        for ivar in range(21):
###           thelist[ivar] = thelist[ivar]-fMin_1[2][ivar]
###           thelist[ivar] = thelist[ivar]*fscale[2][ivar] - 1.
###        solver.test_nets[0].blobs['data'].data[...] = array(thelist)
###        out = solver.test_nets[0].forward(start='ip1')  #don't like out so far
###        #resp[0] = solver.test_nets[0].blobs['fc5'].data
###        respm[0] = solver.test_nets[0].blobs['lp_fc8'].data[0][0]
###        resptm[0] = solver.test_nets[0].blobs['lp_fc8'].data[0][1]
###        outtree.Fill()
###    of.WriteTObject(outtree)
###    of.Close()
###
###
###
###
###
###   
###
###    #print "passes = ",passes, "\t\tfails = ",fails
