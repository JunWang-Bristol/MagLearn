import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class LSTMSeq2One(nn.Module):

    def __init__(self,
                 hidden_size,
                 lstm_num_layers=1,
                 input_size=1,
                 output_size=1):
        super(LSTMSeq2One, self).__init__()

        self.hidden_size = hidden_size
        # LSTM layer
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size+2, 128)
        self.fc2 = nn.Linear(128, 196)
        self.fc3 = nn.Linear(196, 128)
        self.fc4 = nn.Linear(128, 96)
        self.fc5 = nn.Linear(96, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, output_size)

        # Activation function
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        # Default Standardisation Coefficients
        self.std_b = (1.0, 0.0)
        self.std_freq = (1.0, 0.0)
        self.std_loss = (1.0, 0.0)
        self.std_temp = (1.0, 0.0)


    def forward(self, x):

        with torch.no_grad():
            in_b=x[:,:,0:1]
            in_freq=x[:,0,1]
            in_temp=x[:,0,2]

        # Random wave shift
        batch_size, seq_len, _ = in_b.shape
        rand_shifts = torch.randint(seq_len, (batch_size, 1, 1), device=x.device)
        in_b = torch.cat([in_b[i].roll(shifts=int(rand_shifts[i].item()), dims=0).unsqueeze(0) for i in range(batch_size)], dim=0)

        # Random flip vertically and horizontally
        vertical_flip_mask = torch.rand(batch_size, 1, 1, device=x.device) > 0.5
        in_b = torch.where(vertical_flip_mask, -in_b, in_b)

        # lstm layer
        out, _ = self.lstm(in_b)
        out = out[:, -1, :]  # get the last output of the sequence
        

        out_wave = torch.zeros(out.shape[0], out.shape[1] + 2, device=x.device)
        out_wave[:, 0:out.shape[1]] = out
        out_wave[:, -2] = in_freq
        out_wave[:, -1] = in_temp
        out=out_wave


        # fully connected layers
        out = (self.fc1(out))
        out = self.elu(self.fc2(out)) # sigmoid for output between 0 and 1
        out = self.elu(self.fc3(out))
        out = self.elu(self.fc4(out))
        out = self.elu(self.fc5(out))
        out = self.elu(self.fc6(out))
        out = (self.fc7(out))
        out = self.fc8(out)


        # # plot all in_b
        # for i in range(in_b.shape[0]):
        #     plt.figure()
        #     plt.plot(in_b[i].cpu().numpy())
        #     plt.show()


        return out


    def valid(self, x):

        with torch.no_grad():
            in_b=x[:,:,0:1]
            in_freq=x[:,0,1]
            in_temp=x[:,0,2]


        # lstm layer
        out, _ = self.lstm(in_b)
        out = out[:, -1, :]  # get the last output of the sequence
        

        out_wave = torch.zeros(out.shape[0], out.shape[1] + 2, device=x.device)
        out_wave[:, 0:out.shape[1]] = out
        out_wave[:, -2] = in_freq
        out_wave[:, -1] = in_temp
        out=out_wave


        # fully connected layers
        out = (self.fc1(out))
        out = self.elu(self.fc2(out)) # sigmoid for output between 0 and 1
        out = self.elu(self.fc3(out))
        out = self.elu(self.fc4(out))
        out = self.elu(self.fc5(out))
        out = self.elu(self.fc6(out))
        out = (self.fc7(out))
        out = self.fc8(out)


        # # plot all in_b
        # for i in range(in_b.shape[0]):
        #     plt.figure()
        #     plt.plot(in_b[i].cpu().numpy())
        #     plt.show()
        return out
    
    def state_dict(self, *args, **kwargs):
        # Get the base state dict from nn.Module
        base_state_dict = super().state_dict(*args, **kwargs)
        base_state_dict['std_b'] = self.std_b
        base_state_dict['std_freq'] = self.std_freq
        base_state_dict['std_loss'] = self.std_loss
        base_state_dict['std_temp'] = self.std_temp
        return base_state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Extract custom variables from the state dict
        self.std_b = state_dict.pop('std_b', (1.0, 0.0))
        self.std_freq = state_dict.pop('std_freq', (1.0, 0.0))
        self.std_loss = state_dict.pop('std_loss', (1.0, 0.0))
        self.std_temp = state_dict.pop('std_temp', (1.0, 0.0))
        # Load the remaining state dict into the model
        super().load_state_dict(state_dict, strict)


def get_global_model():
    return LSTMSeq2One(hidden_size=30,
                       lstm_num_layers=3,
                       input_size=1,
                       output_size=1)


# Define pytorch loss function to be relative error L=((y-y')/y)^2
class RelativeLoss(nn.Module):

    def __init__(self):
        super(RelativeLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.pow((target - output) / target,2))


#   Define pytorch loss function to be relative error L=|(y-y')/y|
class RelativeLoss_abs(nn.Module):

    def __init__(self):
        super(RelativeLoss_abs, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.abs((target - output) / target))


# Define pytorch loss function to be relative error L=((y-y')/y)^2
class RelativeLoss_95(nn.Module):

    def __init__(self):
        super(RelativeLoss_95, self).__init__()

    def forward(self, output, target):
        error=torch.pow((target - output) / target,2)
        # get best 97% of the data
        error,_=torch.sort(error)
        error=error[0:int(error.shape[0]*0.97)]

        return torch.mean(error)


if __name__ == '__main__':
    # Instantiate the model with appropriate dimensions
    # model = LSTMSeq2One(input_size=3, hidden_size=50, output_size=1)
    model = get_global_model()

    waveStep=128

    # Now we can pass a batch of sequences through the model
    inputs = torch.zeros(
        64, waveStep, 3)  # batch_size = 64, sequence_length = 128, input_size = 3

    wave=torch.linspace(0, 127, waveStep)

    inputs[:, :, 0] = wave
    inputs[:, :, 1] = 10
    inputs[:, :, 2] = 100


    outputs = model(inputs)

    print(outputs.shape)  # Should output torch.Size([64, 1])

    print(outputs[0].shape )

    total_params = sum(p.numel() for p in model.parameters())
    print('model parameters: ', total_params)
