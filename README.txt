Ulrik Soderstrom
Single Layer Perceptron

This code is a single layer perceptron. The code reads in three files (a7a.dev, a7a.train, and 171.test) from the directory it is run from. The perceptron trains on the training data at the following learning rates: [.00001,.0001,.001,.005,.01,.02,.04,.05,.06,.07,.08,.09,.1,.2,.4,.6,.8,1]. These learning rates are graphed against their resulting accuracies against the testing data set. The validation data set is also used after the training to further improve the model weights. The learning rate used is constant, as requested by the professor. The code is commented to describe each function and step. 

The final graph (commented out) displays the testing set under the learning rates to display the optimal values. In this case, often around .09, with an accuracy of ~0.8. Generally, the validation set should be used to optimize, but in this case the testing set is shown against each learning rate. The second graph function (commented out) will graph the learning rate of the model. 


To run this code, use python 2 (or python 3) in a directory with the data files. 