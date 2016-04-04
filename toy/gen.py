from ROOT import gRandom
import numpy as np
import h5py

Nevts = 1000000

train_mc_input_variables = []
train_mc_label = []
test_mc_input_variables = []
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
            test_mc_input_variables.append([x_1,x_2,x_3])
            test_mc_label.append(s_or_b)
    else:
        if train_or_test == 0:
            train_data_input_variables.append([x_1,x_2,x_3])
        else:
            test_data_input_variables.append([x_1,x_2,x_3])
            test_data_label.append(s_or_b)

with h5py.File("train.h5","w") as f:
    f['data'] = np.array(train_mc_input_variables).astype(np.float32)
    f['label'] = np.array(train_mc_label).astype(np.float32)
with h5py.File("grltrain.h5","w") as f:
    f['mc_data'] = np.array(train_mc_input_variables).astype(np.float32)
    f['label'] = np.array(train_mc_label).astype(np.float32)
with h5py.File("test.h5","w") as f:
    f['data'] = np.array(test_mc_input_variables).astype(np.float32)
    f['label'] = np.array(test_mc_label).astype(np.float32)
with h5py.File("data.h5","w") as f:
    f['realdata'] = np.array(train_data_input_variables).astype(np.float32)

