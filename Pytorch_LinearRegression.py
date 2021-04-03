import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)

lr = 0.05  # Learning Rate
n = 30  # number of data points

# Create Data Set
x = torch.rand(n, 1) * 10  # x data (tensor), shape=(100, 1), range from 0 to 10
y = 2 * x + (5 + torch.randn(n, 1))  # y data (tensor), shape=(20, 1), slope = 2, interception = 5, noise is random

x0 = torch.ones(n)  # Create Design Matrix, column 1 as Bias
x0 = torch.reshape(x0, (-1, 1))  # reshape to column vector

x = torch.cat((x0, x), dim=1)  # Combine as design matrix, shape is [n,2]

# Define the parameters for regression
w = torch.rand((2, 1), requires_grad=True)  # 2 parameters: bias and weight, arrange as column vector, shape is [2,1]

for i in range(600):
    y_pred = torch.matmul(x, w)

    loss = (0.5 * (y - y_pred) ** 2).mean()  # loss Function, MSE

    # Back Propagation
    loss.backward()

    # Gradient Descent
    w.data.sub_(lr * w.grad)
    w.grad.zero_()

# Plot
    if i % 30 == 0:
        plt.scatter(x.data[:, 1].numpy(), y.data.numpy())
        plt.plot(x.data[:, 1].numpy(), y_pred.data.numpy(), 'r', lw=4)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {}".format(i, w.data.numpy()))
        plt.pause(0.01)

        if loss.data.numpy() < 0.4:
            break
    plt.show()
