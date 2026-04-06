import sys

import torch.nn as nn
from autoencoder import _Autoencoder
from data import Data
from model import Model
from ae1 import AE1


class AE2(_Autoencoder):

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        self.encoder = Model(
            input_shape=(self.BATCH_SIZE, n_kernels, 32, 32), #input shape is ae1 encoder's output (64, 32, 32)
            layers=[
                nn.Conv2d(n_kernels, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2) #32x32-> 16x16
            ]
        )

        self.decoder = Model(
            input_shape=(self.BATCH_SIZE, n_kernels, 16, 16),
            layers=[
                nn.ConvTranspose2d(n_kernels, n_kernels, kernel_size=2, stride=2), #back to 32x32
                nn.ReLU()
            ]
        )

        self.model = Model(
            input_shape=self.encoder.input_shape,
            layers=[
                self.encoder,
                self.decoder
            ]
        )


if __name__ == '__main__':

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ae1 = AE1('models/ae1.pt')
    data = Data.load('data', image_size=64)
    data = ae1.encode(data) #using ae1's encoded data as input for ae2
    data.shuffle()

    ae = AE2('models/ae2.pt')
    ae.print()

    if not epochs:
        print(f'\nLoading {ae.path}...')
        ae.load()
    else:
        print(f'\nTraining...')
        ae.train(epochs, data)
        print(f'\nSaving {ae.path}...')
        ae.save()

    print(f'\nGenerating samples...')
    samples = ae.generate(data)
    data.display(32)
    samples.display(32)
