![transfer learning](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Knowledge_transfer.svg/312px-Knowledge_transfer.svg.png)

# transfer learning

an attempt to use arXiv:1409.7495 in data from https://www.kaggle.com/c/flavours-of-physics

## transfer-learning / domain adaptation / gradient reversal layer


The authors of [arXiv:1409.7495](http://arxiv.org/abs/1409.7495) show their
idea of how to train a machine learning classifier on some one domain of data
(studio photos / simulated events in HEP) and apply it to another domain of
data (smartphone photos / real data in HEP). The framework is
[Caffe](http://caffe.berkeleyvision.org/) with their fork on
[github](https://github.com/ddtm/caffe/tree/grl).

## flavours of physics

for obvious reasons I have access to simulated Ds→φ(µµ)π events, simulated
background events to Ds→φ(µµ)π (i.e. the simulated events used in the training
of the classifier for τ→µµµ events in
[arXiv:1409.8548](http://arxiv.org/abs/1409.8548), but instead of applying the
τ→µµµ selection, I use the Ds→φ(µµ)π selection). Furthermore, I have a mix of
real events with the Ds→φ(µµ)π selection (with some real events, some
background events, and I hardly know which are which).

## pack it together

I want to do the following

 - train a TMVA classifier (here I know what I'm doing) purely on MC (signal
   and background) and select in data some events and make a nice invariant
   mass plot

 - train Caffe with only MC events (to see how much performance I gain / lose
   when using a toolkit I don't know), make an invariant mass plot with the
   same number of events (see how much cleaner the S/B gets).

 - train Caffe with GRL (i.e. labelled MC events and unlabelled data events).
   Make an invariant mass plot with the same number of events (see how much
   cleaner the S/B gets). In comparison to the other Caffe network, I will see
   how much the transferlearning gains in performance.


# license

The project code is licensed under the [MIT license](LICENSE).

The project logo is from
[wikipedia](https://commons.wikimedia.org/wiki/File:Knowledge_transfer.svgy)
and licensed under [CC0
1.0](https://creativecommons.org/publicdomain/zero/1.0/deed.en).
