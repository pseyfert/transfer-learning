# -*- coding: utf8 -*-
grl = False
m_or_d = 0
import numpy as np
from progressbar import ProgressBar
from ROOT import gRandom

Nevts = 100000
scores = []

for i in xrange(2):
    gRandom.SetSeed(i)
    backgrounds = []
    signals = []
    
    
    for j in xrange(Nevts):
        s_or_b = gRandom.Binomial(1,0.5)
        train_or_test = gRandom.Binomial(1,0.5)
        if s_or_b == 0:
            # x_1 is perfectly simulated, with poor resolution
            x_1 = gRandom.Gaus(-0.5,2.2)
            if m_or_d == 0:
                x_2 = gRandom.Gaus(-1.0,0.2)
                x_3 = gRandom.Gaus(x_2,1.0)
            else:
                x_2 = gRandom.Gaus(+1.0,0.2)
                x_3 = gRandom.Gaus(x_2,1.3)
            backgrounds.append([x_1,x_2,x_3])
        else:
            x_1 = gRandom.Gaus(+0.5,2.2)
            if m_or_d == 0:
                x_2 = gRandom.Gaus(-1.0,0.2)
                x_3 = gRandom.Gaus(x_2+0.5,1.3)
            else:
                x_2 = gRandom.Gaus(+1.0,0.2)
                x_3 = gRandom.Gaus(x_2+0.5,1.3)
    
        
            signals.append([x_1,x_2,x_3])
    
    
    import sys
    sys.path.insert(0,'/home/pseyfert/coding/caffe/python')
    import caffe
        
    caffe.set_mode_cpu()
    if not grl:
      solver = caffe.get_solver("solver.prototxt")
    else:
      solver = caffe.get_solver("grlsolver.prototxt")
    
    niter=9000
    losses = np.zeros(niter+1)
    dlosses = np.zeros(niter+1)
    
    if grl:
      solver.test_nets[0].copy_from('grlsnapshot_iter_'+str(niter)+'.caffemodel')
    else:
      solver.test_nets[0].copy_from('standardsnapshot_iter_'+str(niter)+'.caffemodel')
    
    signal_responses = []
    for e in ProgressBar()(xrange( len(signals))):
    
        solver.test_nets[0].blobs['data'].data[...] = np.array(signals[e])
        out = solver.test_nets[0].forward(start='ip1')  #don't like out so far
        #resp[0] = solver.test_nets[0].blobs['fc5'].data
        resp1 = solver.test_nets[0].blobs['lp_fc8'].data[0][0]
        resp2 = solver.test_nets[0].blobs['lp_fc8'].data[0][1]
        signal_responses.append(resp2)
        
    background_responses = []
    for e in ProgressBar()(xrange( len(backgrounds))):
        solver.test_nets[0].blobs['data'].data[...] = np.array(backgrounds[e])
        out = solver.test_nets[0].forward(start='ip1')  #don't like out so far
        #resp[0] = solver.test_nets[0].blobs['fc5'].data
        resp1 = solver.test_nets[0].blobs['lp_fc8'].data[0][0]
        resp2 = solver.test_nets[0].blobs['lp_fc8'].data[0][1]
        background_responses.append(resp2)
        
    from matplotlib import pylab as plt
    import matplotlib
    matplotlib.rc_file("../lhcb-matplotlibrc/matplotlibrc")
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    ax.hist(signal_responses,range=[0,1],alpha=0.5,color='blue')
    ax.hist(background_responses,range=[0,1],alpha=0.5,color='red')
    
    
    from sklearn.metrics import roc_curve, auc
    resps = signal_responses+background_responses
    one = [1 for s in signal_responses]
    zero = [0 for b in background_responses]
    trues = one+zero
    fpr, tpr,_ = roc_curve(trues,resps)
    score = auc(fpr,tpr)
    scores.append(score)
    if i == 1:
        from ROOT import TFile, TH1F
        f = TFile("responses_"+str(m_or_d)+"_"+str(grl)+".root","recreate")
        resps = TH1F("resps","resps",500,0,1)
        for r in signal_responses:
            resps.Fill(r)
        f.WriteTObject(resps)
        f.Close()

m = np.mean(np.array(scores))
s = np.std(np.array(scores))
from math import sqrt
print u"score is ",m,u"±",s/sqrt(20)
