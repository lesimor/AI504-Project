import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            ############################
            # Define your own discriminator #
            ############################
            nn.Linear(64 * 64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
            ############################
        )

    def forward(self, input):
        #####################################
        # Change the shape of output if necessary #
        input = input.view(-1, 64 * 64)
        #####################################

        output = self.main(input)

        #####################################
        # Change the shape of output if necessary # (batch_size, 1) -> (batch_size, )
        output = output.squeeze(dim=1)
        #####################################

        return output
