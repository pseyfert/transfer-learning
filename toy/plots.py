from ROOT import gRandom
import numpy as np
import h5py

Nevts = 1000000

train_mc_input_variables = []
train_mc_label = []
test_data_input_variables_s = []
test_data_input_variables_b = []
test_mc_input_variables_s = []
test_mc_input_variables_b = []
test_mc_label = []

train_data_input_variables = []
test_data_input_variables = []
test_data_label = []


for i in xrange(Nevts):
    s_or_b = gRandom.Binomial(1,0.5)
    m_or_d = gRandom.Binomial(1,0.5)
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
    else:
        x_1 = gRandom.Gaus(+0.5,2.2)
        if m_or_d == 0:
            x_2 = gRandom.Gaus(-1.0,0.2)
            x_3 = gRandom.Gaus(x_2+0.5,1.3)
        else:
            x_2 = gRandom.Gaus(+1.0,0.2)
            x_3 = gRandom.Gaus(x_2+0.5,1.3)

    if m_or_d == 0:
        if train_or_test == 0:
            train_mc_input_variables.append([x_1,x_2,x_3])
            train_mc_label.append(s_or_b)
        else:
            if s_or_b == 1:
                test_mc_input_variables_s.append([x_1,x_2,x_3])
            else:
                test_mc_input_variables_b.append([x_1,x_2,x_3])
            test_mc_label.append(s_or_b)
    else:
        if train_or_test == 0:
            train_data_input_variables.append([x_1,x_2,x_3])
        else:
            if s_or_b == 1:
                test_data_input_variables_s.append([x_1,x_2,x_3])
            else:
                test_data_input_variables_b.append([x_1,x_2,x_3])
            test_data_label.append(s_or_b)

import matplotlib.pylab as plt

import matplotlib
matplotlib.rc_file("../lhcb-matplotlibrc/matplotlibrc")
fig1, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
axs1.hist(np.array(test_mc_input_variables_s)[:,0],label="signal",range=[-5,5],alpha=0.5,bins=50)
axs1.hist(np.array(test_mc_input_variables_b)[:,0],label="background",range=[-5,5],alpha=0.5,bins=50)
plt.xlabel("x1")
plt.title("distinguishes signal/background (perfectly simulated)")
plt.legend(loc="best")
fig1.show()


fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
axs2.hist(np.array(test_mc_input_variables_s)[:,1],label="MC",range=[-5,5],alpha=0.5,bins=50)
#axs2.hist(np.array(test_mc_input_variables_b)[:,1],label="background MC",range=[-5,5],alpha=0.5,bins=50)
axs2.hist(np.array(test_data_input_variables_s)[:,1],label="DATA",range=[-5,5],alpha=0.5,bins=50)
#axs2.hist(np.array(test_data_input_variables_b)[:,1],label="background DATA",range=[-5,5],alpha=0.5,bins=50)
plt.xlabel("x2")
plt.title("distinguishes MC/DATA")
plt.legend(loc="best")
fig2.show()


fig3, axs3 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
axs3.scatter(np.array(test_mc_input_variables_s)[:100,1],np.array(test_mc_input_variables_s)[:100,2],label="signal MC",c="blue")
axs3.scatter(np.array(test_mc_input_variables_b)[:100,1],np.array(test_mc_input_variables_b)[:100,2],label="background MC",c="red")
axs3.scatter(np.array(test_data_input_variables_s)[:100,1],np.array(test_data_input_variables_s)[:100,2],label="signal DATA",c="cyan")
axs3.scatter(np.array(test_data_input_variables_b)[:100,1],np.array(test_data_input_variables_b)[:100,2],label="background DATA",c="magenta")
plt.xlabel("x2")
plt.ylabel("x3")
plt.legend(loc="best")
fig3.show()



fig4, axs4 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
axs4.hist(np.array(test_mc_input_variables_s)[:,2],label="signal MC",color='blue',range=[-5,5],alpha=0.5,bins=50)
axs4.hist(np.array(test_mc_input_variables_b)[:,2],label="background MC",color='red',range=[-5,5],alpha=0.5,bins=50)
axs4.hist(np.array(test_data_input_variables_s)[:,2],label="signal DATA",color='cyan',range=[-5,5],alpha=0.5,bins=50)
axs4.hist(np.array(test_data_input_variables_b)[:,2],label="background DATA",color='magenta',range=[-5,5],alpha=0.5,bins=50)
plt.legend(loc="best")
plt.xlabel("x3")
plt.title("distinguishes signal/background (poorly simulated)")
fig4.show()
