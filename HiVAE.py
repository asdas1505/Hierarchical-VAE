import numpy as np
import torch
import torch.nn as nn
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


bs = 1024
# MNIST Dataset
decoder_distribution = 'Normal'

if decoder_distribution=='Normal':
    print('Transformation for normally distributed decoder')
    transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    # transform = transforms.ToTensor()
elif decoder_distribution=='Bernoulli':
    print('Transformation for bernoulli distributed decoder')
    transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_parameters(model):
    latent = True
    encoder_params = []
    decoder_params = []
    for param in model.parameters():
        if len(param.shape) == 2:
            if param.shape[1] == model.param_split:
                latent = False
        if latent:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    all_params = encoder_params + decoder_params

    return all_params, encoder_params, decoder_params

def grad_switch(params, grad_bool=False):
    for param in params:
        param.requires_grad = grad_bool


class PHI(nn.Module):
    def __init__(self):
        super(PHI, self).__init__()
        self.mu_e1_p = nn.Parameter()
        self.logvar_e1_p = nn.Parameter()

        self.mu_e2_p = nn.Parameter()
        self.logvar_e2_p = nn.Parameter()

        self.mu_e1 = 0
        self.logvar_e1 = 0

        self.mu_e2 = 0
        self.logvar_e2 = 0


class iHVAE(nn.Module):
    def __init__(self, z2_dim=10, z1_dim=20, n_iters=20, eval_iters=500, decoder_distribution='Normal'):
        super(iHVAE, self).__init__()

        self.decoder_distribution = decoder_distribution

        self.latent_space_size1 = z1_dim
        self.latent_space_size2 = z2_dim

        self.param_split = z2_dim

        self.n_iters = n_iters
        self.eval_iters = eval_iters

        ## Encoder 1
        self.e1fc1 = nn.Linear(784, 512)
        self.e1fc2 = nn.Linear(512, 256)
        ## mu1 and logvar1
        self.e1fc4 = nn.Linear(256, z1_dim)
        self.e1fc5 = nn.Linear(256, z1_dim)

        ## Encoder 2
        self.e2fc1 = nn.Linear(z1_dim, 256)
        self.e2fc2 = nn.Linear(256, 128)
        ## mu2 and logvar2
        self.e2fc4 = nn.Linear(128, z2_dim)
        self.e2fc5 = nn.Linear(128, z2_dim)

        ## Decoder 2
        self.d2fc1 = nn.Linear(z2_dim, 128)
        self.d2fc2 = nn.Linear(128, 256)
        ## d_mu1 and d_logvar1
        self.d2fc4 = nn.Linear(256, z1_dim)
        self.d2fc5 = nn.Linear(256, z1_dim)

        ## Decoder 1
        self.d1fc1 = nn.Linear(z1_dim, 256)
        self.d1fc2 = nn.Linear(256, 512)
        self.d1fc4 = nn.Linear(512, 784)

        self.param_enc = [self.e1fc1.weight, self.e1fc1.bias, self.e1fc2.weight, self.e1fc2.bias,
                          self.e1fc4.weight, self.e1fc4.bias, self.e1fc5.weight, self.e1fc5.bias,
                          self.e2fc1.weight, self.e2fc1.bias, self.e2fc2.weight, self.e2fc2.bias,
                          self.e2fc4.weight, self.e2fc4.bias, self.e2fc5.weight, self.e2fc5.bias ]

        self.param_dec = [self.d2fc1.weight, self.d2fc1.bias, self.d2fc2.weight, self.d2fc2.bias,
                          self.d2fc4.weight, self.d2fc4.bias, self.d2fc5.weight, self.d2fc5.bias,
                          self.d1fc1.weight, self.d1fc1.bias, self.d1fc2.weight, self.d1fc2.bias,
                          self.d1fc4.weight, self.d1fc4.bias ]

    def encoder1(self, x):
        x = torch.tanh(self.e1fc1(x))
        x = torch.tanh(self.e1fc2(x))

        mu1 = self.e1fc4(x)
        logvar1 = self.e1fc5(x)

        return mu1, logvar1

    def encoder2(self, z2):
        z2 = torch.tanh(self.e2fc1(z2))
        z2 = torch.tanh(self.e2fc2(z2))

        mu2 = self.e2fc4(z2)
        logvar2 = self.e2fc5(z2)

        return mu2, logvar2

    def decoder2(self, z2):
        z1_rec = torch.tanh(self.d2fc1(z2))
        z1_rec = torch.tanh(self.d2fc2(z1_rec))

        d_mu1 = self.d2fc4(z1_rec)
        d_logvar1 = self.d2fc5(z1_rec)

        return d_mu1, d_logvar1

    def decoder1(self, z1):
        x_rec = torch.tanh(self.d1fc1(z1))
        x_rec = torch.tanh(self.d1fc2(x_rec))
        x_rec = self.d1fc4(x_rec)
        return x_rec

    def r_sampling(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        var = torch.exp(0.5 * logvar)
        z = var.mul(epsilon).add(mu)
        return z, epsilon

    def forward(self, x):
        phi = PHI()

        phi.mu_e1_p.data, phi.logvar_e1_p.data = self.encoder1(x)

        optim_infer = torch.optim.Adam(phi.parameters(), lr=1e-2)
        inference_params = list(phi.parameters())

        grad_switch(inference_params, grad_bool=True)
        for infer_iter in range(self.n_iters):
            optim_infer.zero_grad()
            _, _, _, total_loss = self.iter_inference(x, phi)
            optim_infer.step()
        grad_switch(inference_params, grad_bool=False)

        return phi

    def forward_evaluate(self, x, x_clear=None):
        phi = PHI()

        phi.mu_e1_p.data, phi.logvar_e1_p.data = self.encoder1(x)

        optim_infer = torch.optim.Adam(phi.parameters(), lr=1e-3)
        inference_params = list(phi.parameters())

        grad_switch(inference_params, grad_bool=True)
        for infer_iter in range(self.eval_iters):
            optim_infer.zero_grad()
            _, _, _, total_loss = self.iter_inference(x, phi, x_clear)
            optim_infer.step()
        grad_switch(inference_params, grad_bool=False)

        return phi

    def iter_inference(self, x, phi=None, mu=None, logvar=None, decoder=True, x_clear=None, evaluate=False):
        if phi is not None:
            z1_sample, e_eps1 = self.r_sampling(phi.mu_e1_p, phi.logvar_e1_p)
            phi.mu_e2_p.data, phi.logvar_e2_p.data = self.encoder2(z1_sample)
            z2_sample, e_eps2 = self.r_sampling(phi.mu_e2_p, phi.logvar_e2_p)
            mu_d1, logvar_d1 = self.decoder2(z2_sample)
            z1_rec_sample, d_eps1 = self.r_sampling(mu_d1, logvar_d1)
            x_rec = self.decoder1(z1_sample)
            z2_var_params = (phi.mu_e2_p, phi.logvar_e2_p, e_eps2)
            z1_var_params = (phi.mu_e1_p, phi.logvar_e1_p, e_eps1)
            d_z1_var_params = (mu_d1, logvar_d1, d_eps1)
            total_loss = hvae_loss(x, x_rec, z2_var_params, z1_var_params, d_z1_var_params, x_clear)
        else:
            z1_sample, e_eps1 = self.r_sampling(mu, logvar)
            mu2, logvar2 = self.encoder2(z1_sample)
            z2_sample, e_eps2 = self.r_sampling(mu2, logvar2)
            mu_d1, logvar_d1 = self.decoder2(z2_sample)
            z1_rec_sample, d_eps1 = self.r_sampling(mu_d1, logvar_d1)
            x_rec = self.decoder1(z1_sample)
            z2_var_params = (mu2, logvar2, e_eps2)
            z1_var_params = (mu, logvar, e_eps1)
            d_z1_var_params = (mu_d1, logvar_d1, d_eps1)
            total_loss = hvae_loss(x, x_rec, z2_var_params, z1_var_params, d_z1_var_params, x_clear)

        total_loss.backward()

        return x_rec, z1_sample, z2_sample, total_loss


def hvae_loss(x, x_rec, z2_var_params, z1_var_params, d_z1_var_params, x_clear=None, decoder_distribution='Normal'):
    """
    Working:
    Expanding the log of Normal distribution and expanding it in terms of z1 and z2
    """
    if decoder_distribution=='Normal':
        if x_clear is not None:
            REC_ERROR = F.mse_loss(x_rec, x_clear, reduction='sum')
        else:
            REC_ERROR = F.mse_loss(x_rec, x, reduction='sum')
    elif decoder_distribution=='Bernoulli':
        REC_ERROR = F.binary_cross_entropy(torch.sigmoid(x_rec), x, reduction='sum')

    mu2, logvar2, epsilon2 = z2_var_params
    mu1, logvar1, epsilon1 = z1_var_params
    d_mu1, d_logvar1, d_epsilon1 = d_z1_var_params

    log_q_z1_x = torch.sum(-0.5 * (epsilon1 ** 2) - 0.5 * logvar1, dim=-1)
    log_q_z2_z1 = torch.sum(-0.5 * (epsilon2 ** 2) - 0.5 * logvar2, dim=-1)

    z2 = torch.exp(0.5 * logvar2).mul(epsilon2).add(mu2)
    z1 = torch.exp(0.5 * logvar1).mul(epsilon1).add(mu1)
    dvar1 = torch.exp(0.5 * d_logvar1)

    log_p_z2 = torch.sum(-0.5 * (z2 ** 2), dim=-1)
    log_p_z1_z2 = torch.sum(-0.5 * (((z1 - d_mu1) / dvar1) ** 2) - 0.5 * d_logvar1, dim=-1)

    ERROR_TERM2 = torch.mean(torch.sum(log_q_z1_x - log_p_z1_z2, axis=0))
    ERROR_TERM3 = torch.mean(torch.sum(log_q_z2_z1 - log_p_z2, axis=0))

    return REC_ERROR + ERROR_TERM2 + ERROR_TERM3


def evaluate(model, data):
    model.eval()

    phi = model(data)

    z1_sample, e_eps1 = model.r_sampling(phi.mu_e1_p, phi.logvar_e1_p)
    z2_sample, e_eps2 = model.r_sampling(phi.mu_e2_p, phi.logvar_e2_p)
    mu_d1, logvar_d1 = model.decoder2(z2_sample)
    d_z1_sample, d_eps1 = model.r_sampling(mu_d1, logvar_d1)

    x_rec = model.decoder1(z1_sample)

    z2_var_params = (phi.mu_e2_p, phi.logvar_e2_p, e_eps2)
    z1_var_params = (phi.mu_e1_p, phi.logvar_e1_p, e_eps1)
    d_z1_var_params = (mu_d1, logvar_d1, d_eps1)

    total_loss = hvae_loss(data, x_rec, z2_var_params, z1_var_params, d_z1_var_params)

    return total_loss


def train_iHVAE(epoch):
    model.train()
    total_loss = 0

    all_params = model.param_enc + model.param_dec
    for idx, (X_train, y_train) in enumerate(train_loader):
        grad_switch(all_params, grad_bool=False)

        X_train = torch.flatten(X_train, start_dim=1).to(device)
        phi = model(X_train)

        # Updating decoding paramters
        grad_switch(model.param_dec, grad_bool=True)
        optimizer_iHVAE.zero_grad()
        model.iter_inference(X_train, phi)
        optimizer_iHVAE.step()
        grad_switch(model.param_dec, grad_bool=False)

        # Updating encoding paramters
        grad_switch(model.param_enc, grad_bool=True)
        optimizer_iHVAE.zero_grad()
        mu1, logvar1 = model.encoder1(X_train)
        model.iter_inference(X_train, phi=None, mu=mu1, logvar=logvar1, decoder=False)
        optimizer_iHVAE.step()
        grad_switch(model.param_enc, grad_bool=False)

        loss = evaluate(model, X_train)
        total_loss += loss
    total_loss /= (len(train_loader.dataset))
    print('For epoch {}, loss is {}'.format(epoch + 1, total_loss))


def reconstruct_images(model, data, x_clear=None):
    model.eval()
    phi = model.forward_evaluate(data, x_clear)
    z1_sample, _ = model.r_sampling(phi.mu_e1_p, phi.logvar_e1_p)
    x_rec = model.decoder1(z1_sample)

    return x_rec

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

z1 = 100
z2 = 50
model = iHVAE(z1_dim=z1, z2_dim=z2).to(device)
optimizer_iHVAE = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 500
for e in range(epochs):
    train_iHVAE(e)

PATH = 'HiVAE-epochs-{}-z1-{}-z2-{}-decoder-{}.pth'.format(epochs, z1, z2, model.decoder_distribution)
torch.save(model.state_dict(), PATH)


model.load_state_dict(torch.load(PATH))

classifier = Classifier().to(device)
PATH_Classifier = 'mnist-classifier-epochs-40.pth'
classifier.load_state_dict(torch.load(PATH_Classifier))

def gaussian_noise(input, std=0.0):
    noise = torch.normal(mean=0, std=torch.ones_like(input)*std)
    return input + noise

noise_stds = [0, 0.2, 0.4, 0.6, 0.8]


for std in noise_stds:
    correct = 0
    for idx, (X_test, target) in enumerate(test_loader):
        X_test = X_test.to(device).view(-1, 784)
        target = target.to(device)
        X_test_noisy = gaussian_noise(X_test, std=std)
        X_test_noisy = (X_test_noisy - 0.1307) / 0.3081
        X_test = (X_test - 0.1307) / 0.3081
        X_test_rec = reconstruct_images(model, X_test_noisy, X_test)
        X_test_rec = X_test_rec.view(-1, 1, 28, 28)
        output = classifier(X_test_rec)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    mean_accuracy = 100. * correct / len(test_loader.dataset)
    print('White Noise with std = {}, accuracy is {}'.format(std, mean_accuracy))