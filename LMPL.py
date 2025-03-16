

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter



# get indices   (can be random)

# func used to calculate J
class My_SoftPlus:
    def __init__(self, alpha, beta=10):
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        f = torch.nn.Softplus(beta=self.beta)
        x = x + f(self.alpha - x)
        return x

    # def backward(self, x):
    #     x = self.beta * (x - self.alpha)
    #     x = torch.exp(x)
    #     x = 1 / (1 + x)
    #     return x

    def __call__(self, x):
        return self.forward(x)


def g(alpha, x):
    return My_SoftPlus(alpha)(x)


class My_SoftPlus_F:
    def __init__(self, alpha, beta=10):
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        f = torch.nn.Softplus(beta=self.beta)
        x = x - f(x - self.alpha)
        return x

    def backward(self, x):
        x = self.beta * (x - self.alpha)
        x = torch.exp(x)
        x = 1 / (1 + x)
        return x

    def __call__(self, x):
        return self.forward(x)


def f(alpha, x):
    return My_SoftPlus_F(alpha)(x)


def df(alpha, x):
    return My_SoftPlus_F(alpha).backward(x)

class LMPL_F(nn.Module):
    def __init__(self, input_dim, output_dim, alpha, hidden_layers=None):
        super(LMPL_F, self).__init__()
        self.alpha=alpha
        self.bias = nn.Parameter(torch.randn(output_dim))
        self.activation = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        #create hidden layers
        if hidden_layers:
            #input layer
            self.hidden_layers.append(nn.Linear(input_dim, hidden_layers[0]))
            #hidden layers
            for i in range(1,len(hidden_layers)):
                self.hidden_layers.append(nn.Linear(hidden_layers[i-1],hidden_layers[i]))
            #output layer
            self.hidden_layers.append(nn.Linear(hidden_layers[-1],output_dim))
        else:
            self.hidden_layers.append(nn.Linear(input_dim, output_dim))
        #initialize weights
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x=self.activation(layer(x))
        y=self.hidden_layers[-1](x)+self.bias
        return y

    def get_indices(self, data_size):
        indices = torch.triu_indices(data_size, data_size, offset=1)
        indices = indices.index_select(0, torch.LongTensor([1, 0]))
        indices = torch.transpose(indices, 0, 1)
        return indices

    def get_distance(self, x, indices):
        resized_x = x.reshape(1, x.size(0), -1)
        dist_mat = torch.cdist(resized_x, resized_x).squeeze()
        dist = dist_mat[indices[:, 0], indices[:, 1]]
        return dist

    def get_J(self, inputs, outputs, alpha):
        batch_size = inputs.size(0)
        indices = self.get_indices(batch_size)
        dis_in = self.get_distance(inputs, indices)
        dis_out = self.get_distance(outputs, indices)


        a_in = self.alpha
        a_out = self.alpha
        Is = f(a_out, dis_out) - f(a_in, dis_in)
        Js = torch.square(Is)
        mean_J = torch.mean(Js)

        return mean_J

    def backward(self, J):
        J.backward()

    def update_weights(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()


class LMPL(nn.Module): #add dropout
    def __init__(self, input_dim, output_dim, alpha, epsilon, hidden_layers=None, dropout_prob=0.5):
        super(LMPL, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.bias = nn.Parameter(torch.randn(output_dim))
        self.activation = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout layer with configurable probability

        # Create hidden layers
        if hidden_layers:
            self.hidden_layers.append(nn.Linear(input_dim, hidden_layers[0]))
            for i in range(1, len(hidden_layers)):
                self.hidden_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.hidden_layers.append(nn.Linear(hidden_layers[-1], output_dim))
        else:
            self.hidden_layers.append(nn.Linear(input_dim, output_dim))

        # Initialize weights
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)  # Apply dropout after activation
        y = self.hidden_layers[-1](x) + self.bias
        return y

    def get_indices(self, data_size):
        indices = torch.triu_indices(data_size, data_size, offset=1)
        indices = indices.index_select(0, torch.LongTensor([1, 0]))
        indices = torch.transpose(indices, 0, 1)
        return indices

    def get_distance(self, x, indices):
        resized_x = x.reshape(1, x.size(0), -1)
        dist_mat = torch.cdist(resized_x, resized_x).squeeze()
        dist = dist_mat[indices[:, 0], indices[:, 1]]
        return dist

    def get_J(self, inputs, outputs, alpha):
        batch_size = inputs.size(0)
        indices = self.get_indices(batch_size)
        dis_in = self.get_distance(inputs, indices)
        dis_out = self.get_distance(outputs, indices)
        a = self.alpha
        v_in = 1 / (dis_in + self.epsilon)
        v_out = 1 / (dis_out + self.epsilon)
        Is = g(a, v_out) - g(a, v_in) - self.alpha * 0.1
        Js = torch.square(Is)
        mean_J = torch.mean(Js)

        return mean_J

    def backward(self, J):
        J.backward()

    def update_weights(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()


class LMPL_org(nn.Module):
    def __init__(self, input_dim, output_dim, alpha,epsilon, hidden_layers=None):
        super(LMPL_org, self).__init__()
        self.alpha=alpha
        self.epsilon=epsilon
        #self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))
        self.activation = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        #create hidden layers
        if hidden_layers:
            #input layer
            self.hidden_layers.append(nn.Linear(input_dim, hidden_layers[0]))
            #hidden layers
            for i in range(1,len(hidden_layers)):
                self.hidden_layers.append(nn.Linear(hidden_layers[i-1],hidden_layers[i]))
            #output layer
            self.hidden_layers.append(nn.Linear(hidden_layers[-1],output_dim))
        else:
            self.hidden_layers.append(nn.Linear(input_dim, output_dim))
        #initialize weights
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x=self.activation(layer(x))
        y=self.hidden_layers[-1](x)+self.bias
        return y

    def get_indices(self, data_size):
        indices = torch.triu_indices(data_size, data_size, offset=1)
        indices = indices.index_select(0, torch.LongTensor([1, 0]))
        indices = torch.transpose(indices, 0, 1)
        return indices

    def get_distance(self, x, indices):
        resized_x = x.reshape(1, x.size(0), -1)
        dist_mat = torch.cdist(resized_x, resized_x).squeeze()
        dist = dist_mat[indices[:, 0], indices[:, 1]]
        return dist


    def get_J(self, inputs, outputs, alpha):
        batch_size = inputs.size(0)
        indices = self.get_indices(batch_size)
        dis_in = self.get_distance(inputs, indices)
        dis_out = self.get_distance(outputs, indices)
        a = self.alpha
        v_in = 1 / (dis_in + self.epsilon)
        v_out = 1 / (dis_out + self.epsilon)
        Is = g(a, v_out) - g(a, v_in)-self.alpha*0.1
        Js = torch.square(Is)
        mean_J = torch.mean(Js)

        return mean_J

    def backward(self, J):
        J.backward()

    def update_weights(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()

def train_org(model, optimizer, x_train, epochs, alpha,scheduler=None):
    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        outputs = model(x_train)
        loss = model.get_J(x_train, outputs, alpha)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step(loss)
        if (epoch + 1) % 100 == 0 or epoch==0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return outputs


def train_record(model, optimizer, x_train, epochs, alpha,scheduler=None):
    losses=[]
    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        outputs = model(x_train)
        loss = model.get_J(x_train, outputs, alpha)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if scheduler:
            scheduler.step(loss)

        if (epoch + 1) % 100 == 0 or epoch==0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return outputs,losses

def train(model, optimizer, x_train, epochs,alpha,log_dir='log',scheduler=None):
    writer = SummaryWriter(log_dir=log_dir)  # Initialize TensorBoard writer

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = model.get_J(x_train,outputs,alpha)   # loss value for each layer and add them together
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step(loss)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        writer.add_scalar('Loss/train', loss.item(), epoch)

        # 记录参数和梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'{name}.data', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad, epoch)
                #writer.add_histogram(f'{name}.grad', param.grad, epoch)

    writer.close()
    return outputs

def train_layerloss(model, optimizer, x_train, epochs,log_dir='log'):
    writer = SummaryWriter(log_dir=log_dir)  # Initialize TensorBoard writer

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs,layer_outputs = model(x_train)
        loss = model.compute_loss_layer(x_train, layer_outputs)   # loss value for each layer and add them together
        model.backward(loss)
        model.update_weights(optimizer)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        writer.add_scalar('Loss/train', loss.item(), epoch)

        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'{name}.data', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad, epoch)
                #writer.add_histogram(f'{name}.grad', param.grad, epoch)

    writer.close()
    return outputs


