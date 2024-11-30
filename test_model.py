import sys, os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.pyplot import axes
import torch
import torch.nn as nn
import tqdm
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import ann02db02 as db  # имена файлов данных


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class PudgeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc6   = nn.Linear(32, 4)
        self.test1 = nn.Linear(20, 32)
        self.test2 = nn.Linear(32, 32)
        self.test3 = nn.Linear(32, 32)
        self.test4 = nn.Linear(32, 32)
        # self.test5 = nn.Linear(32, 32)
        # self.test6 = nn.Linear(32, 32)
        # self.test7 = nn.Linear(16, 16)
        # self.test8 = nn.Linear(16, 16)
        self.tanh  = nn.Tanh()
        self.relu  = nn.ReLU()

    def forward(self, x):

        x = self.tanh(self.test1(x))
        x = self.tanh(self.test2(x))
        x = self.tanh(self.test3(x))
        x = self.tanh(self.test4(x))
        # x = self.tanh(self.test5(x))
        # x = self.tanh(self.test6(x))
        # x = self.tanh(self.test7(x))
        # x = self.tanh(self.test8(x))
        x = self.fc6(x)

        return x


class PudgeNet_p(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc6   = nn.Linear(32, 4)
        self.test1 = nn.Linear(16, 32)
        self.test2 = nn.Linear(32, 32)
        self.test3 = nn.Linear(32, 32)
        self.test4 = nn.Linear(32, 32)
        # self.test5 = nn.Linear(32, 32)
        # self.test6 = nn.Linear(32, 32)
        # self.test7 = nn.Linear(16, 16)
        # self.test8 = nn.Linear(16, 16)
        self.tanh  = nn.Tanh()
        self.relu  = nn.ReLU()

    def forward(self, x):

        x = self.tanh(self.test1(x))
        x = self.tanh(self.test2(x))
        x = self.tanh(self.test3(x))
        x = self.tanh(self.test4(x))
        # x = self.tanh(self.test5(x))
        # x = self.tanh(self.test6(x))
        # x = self.tanh(self.test7(x))
        # x = self.tanh(self.test8(x))
        x = self.fc6(x)

        return x


def evaluate(model, dataloader, loss_fn):

    # y_pred_list = []
    # y_true_list = []
    losses = []
    model.eval()

    for i, batch in enumerate(dataloader):
        X_batch, y_batch = batch

        with torch.no_grad():
            pred = model(X_batch.to(device))

            loss = loss_fn(pred, y_batch.to(device))
            loss = loss.item()

            losses.append(loss)

            # y_pred = pred

        # y_pred_list.extend(y_pred.cpu().numpy())
        # y_true_list.extend(y_batch.numpy())

    return np.mean(losses)


def train(model, loss_fn, optimizer, train_loader, train_data, val_data, n_epoch=6):
    model.train(True)
    data = {
        'loss_train': [],
        'loss_val': []
    }
    for epoch in tqdm(range(n_epoch)):
        for i, batch in enumerate(train_loader):
            X_batch, y_batch = batch
            pred = model(X_batch.to(device))
            loss = loss_fn(pred, y_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('On epoch end', epoch)
        loss_train_epoch = evaluate(model, train_data, loss_fn)
        # print('Train loss:', loss_train_epoch)
        loss_val_epoch = evaluate(model, val_data, loss_fn)
        # print('Train loss:', loss_train_epoch, 'Val loss:', loss_val_epoch)

        data['loss_train'].append(loss_train_epoch)
        data['loss_val'].append(loss_val_epoch)

    return model, data


def spline_plot(y_true, y_pred, i):
    x_nods = np.array([0., 0.333, 0.666, 1.])
    x = np.linspace(0, 1, 100)
    plt.xlim([0, 1])
    plt.ylim([0, 5])
    error = abs(y_true - y_pred) / y_true

    # bc_l_true = (y_true[1] - y_true[0]) / 0.333
    # bc_l_pred = (y_pred[1] - y_pred[0]) / 0.333
    # bc_r_true = (y_true[3] - y_true[2]) / 0.333
    # bc_r_pred = (y_pred[3] - y_pred[2]) / 0.333

    cs_true = CubicSpline(x_nods, y_true, bc_type=((1, 0), (1, 0)))
    cs_pred = CubicSpline(x_nods, y_pred, bc_type=((1, 0), (1, 0)))
    plt.plot(x, cs_true(x), color='r', label='True')
    plt.plot(x, cs_pred(x), color='b', label='Predicted')
    plt.scatter(x_nods, y_true, color='m')
    plt.scatter(x_nods, y_pred, color='c')
    plt.suptitle(f'{error}')
    plt.legend()
    plt.savefig('D:/Programs/ann0202/graphs/{}.png'.format(i))
    plt.clf()


def save(model, name):
    torch.save(model.state_dict(), f"{name}")


def loadNN(model, name):
    model.load_state_dict(torch.load(f"{name}"))
    model.eval()
    return model


# ************************************************************************************************


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.version.cuda)
print(torch.cuda.is_available())

db.rdbann22()

nsbd = db.dbmt.shape[0]  # количество образцов
batch_size = 64

# number of batches
nbatch  = nsbd // batch_size         # number of total      batches
nbtest  = 16                         # number of test       batches
nbval   = 16                         # number of validation batches
nbtrain = nbatch - nbtest - nbval    # number of train      batches

# number of samples
nstrain = nbtrain * db.batch_size    # number of train      samples
nsval   = nbval   * db.batch_size    # number of validation samples
nstest  = nbtest  * db.batch_size    # number of test       samples

print(nstrain, nsval, nstest)

data1 = db.dbfk
data2 = db.dbsm
data3 = db.dbrn



# data_all = np.hstack((data1, data2, data3))
data_all  = data3[:nstrain + nsval]
data_test = data3[nsbd - nstest:].copy()
samples   = np.load('samples_poisson.npy')[:, 0:4, 0]
poisson   = np.load('samples_poisson.npy')[:, 0:4, 1]

poisson_train = poisson[:nstrain + nsval, :].copy()
poisson_test  = poisson[nstrain + nsval:, :].copy()
# np.save('data_all.npy', data_all)

# data_all  = (data_all  - np.min(data_all))   / (np.max(data_all)  - np.min(data_all))  - 0.5
# data_test = (data_test - np.min(data_test))  / (np.max(data_test) - np.min(data_test)) - 0.5
data_all      = (data_all      - np.mean(data_all))      / np.std(data_all)
data_test     = (data_test     - np.mean(data_test))     / np.std(data_test)
poisson_train = (poisson_train - np.mean(poisson_train)) / np.std(poisson_train)
poisson_test  = (poisson_test  - np.mean(poisson_test))  / np.std(poisson_test)

# data_all  = np.hstack((data_all, poisson_train))
# data_test = np.hstack((data_test, poisson_test))


samples1 = samples[:, 3]
samples2 = samples[:, 2]
samples3 = samples[:, 1]
samples4 = samples[:, 0]

samples1 = np.reshape(samples1, (len(samples1), 1))
samples2 = np.reshape(samples2, (len(samples2), 1))
samples3 = np.reshape(samples3, (len(samples3), 1))
samples4 = np.reshape(samples4, (len(samples4), 1))

# ***********************************************************************************************************

train_data1 = MyDataset(data_all[:nstrain], samples[:nstrain])
val_data1   = MyDataset(data_all[nstrain:nstrain + nsval], samples[nstrain:nstrain + nsval])
test_data1  = MyDataset(data_test, samples[nsbd - nstest:])

train_loader1 = DataLoader(train_data1, batch_size=batch_size, shuffle=True)
val_loader1   = DataLoader(val_data1,   batch_size=batch_size, shuffle=False)
test_loader1  = DataLoader(test_data1,  batch_size=batch_size, shuffle=False)

epochs = 3000

model1 = PudgeNet_p().to(device)
# loss_fn = torch.nn.MSELoss()
loss_fn = MSLELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)

print(sum(p.numel() for p in model1.parameters()))

# model1 = loadNN(model1, 'PudgeNet_test13_final_4_dbrn')
model1, loss1 = train(model1, loss_fn, optimizer, train_loader1, train_data1, val_data1, n_epoch=epochs)
save(model1, 'PudgeNet_test13_final_0_dbrn')

# predict_all1 = model1(torch.tensor(data_all, dtype=torch.float))
# predict_all1 = predict_test1.detach().cpu().numpy()
y_pred = model1(test_data1.X)
y_pred = y_pred.detach().cpu().numpy()
y_true = samples[nsbd - nstest:]
rel_abs_error = abs(y_pred - y_true) / y_true
np.save('error.npy', rel_abs_error)
rel_abs_error = np.mean(rel_abs_error, 0)

fig, axs = plt.subplots(figsize=(10, 5))
ox1 = list(range(epochs))
axs.plot(ox1, loss1['loss_train'], label='Train loss', color='r')
axs.title.set_text('dbrn')
axs.plot(ox1, loss1['loss_val'], label='Val loss', color='m')
axs.legend()

fig.suptitle(f'{rel_abs_error}')
plt.savefig('Loss_dbrn_13_poisson_final_0.png')
# plt.show()
plt.clf()

# fig, axs = plt.subplots(figsize=(7, 5))
# for i in range(len(y_true)):
#     spline_plot(y_true[i], y_pred[i], i)
