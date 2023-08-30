import torch
import torch.nn as nn


class LSTMSeq2One(nn.Module):

    def __init__(self,
                 hidden_size,
                 lstm_num_layers=1,
                 input_size=3,
                 output_size=1):
        super(LSTMSeq2One, self).__init__()

        self.hidden_size = hidden_size
        # LSTM layer
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 8)
        self.fc6 = nn.Linear(8, output_size)

        # Activation function
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU()

    def forward(self, x):

        # lstm layer
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # get the last output of the sequence

        # fully connected layers
        out = self.elu(self.fc1(out))
        out = self.elu(self.fc2(out))
        out = self.elu(self.fc3(out))
        out = self.elu(self.fc4(out))
        out = self.elu(self.fc5(out))
        out = self.fc6(out)

        return out


def get_global_model():
    return LSTMSeq2One(hidden_size=30,
                       lstm_num_layers=1,
                       input_size=3,
                       output_size=1)


# Define pytorch loss function to be relative error L=((y-y')/y)^2
class RelativeLoss(nn.Module):

    def __init__(self):
        super(RelativeLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.pow((target - output) / target), 2)


#   Define pytorch loss function to be relative error L=|(y-y')/y|
class RelativeLoss_abs(nn.Module):

    def __init__(self):
        super(RelativeLoss_abs, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.abs((target - output) / target))

if __name__ == '__main__':
    # Instantiate the model with appropriate dimensions
    # model = LSTMSeq2One(input_size=3, hidden_size=50, output_size=1)
    model = get_global_model()

    # Now we can pass a batch of sequences through the model
    inputs = torch.rand(
        64, 128, 3)  # batch_size = 64, sequence_length = 128, input_size = 3
    outputs = model(inputs)

    print(outputs.shape)  # Should output torch.Size([64, 1])

    print(outputs[0].shape
          )  # Should output tensor([-0.0005], grad_fn=<SelectBackward>
