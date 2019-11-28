import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
input_size = 1
output_size = 1
epochs = 50
learning_rate = 0.0001

# training data
h_train = np.array([[0],[1],[2],[3],[4],[5]],dtype=np.float32)
r_train = np.array([[10],[12],[7],[13],[5],[9]],dtype=np.float32)

# linear regression model
model = nn.Linear(input_size,output_size)

# loss function (mean squared error)
loss_function = nn.MSELoss()
# optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# train the model
for epoch in range(epochs):
	inputs = torch.from_numpy(h_train)
	targets = torch.from_numpy(r_train)

	# forward propagation
	predictions = model(inputs)
	loss = loss_function(predictions,targets)

	# backward propagation
	loss.backward()
	# optimization
	optimizer.step()

	print 'Epoch: ' + str(epoch) + '.....' + 'Loss: ' + str(loss.item())

# plot the best fit line
prediction = model(torch.from_numpy(h_train)).detach().numpy()
plt.plot(h_train, r_train, 'v--g', label='original data')
plt.plot(h_train, prediction, 'r', label='best fit line')
plt.legend()
plt.show()