'''See class AutoEncoderWrapper'''
import torch
import pytorch_lightning as pl
import torchio as tio

__all__ = [
    'AutoEncoderWrapper'
]


class AutoEncoderWrapper(pl.LightningModule):
    '''Simple pytorch lightning wrapper for an autoencoder '''
    def __init__(self, autoencoder, loss, learning_rate, use_sigmoid=True):
        super().__init__()
        self.autoencoder = autoencoder
        self.loss = loss
        self.learning_rate = learning_rate
        self.use_sigmoid = use_sigmoid
        self.save_hyperparameters() # Loading from checkpoint does not work if have ignore=['loss']

    def forward(self, x):
        # pylint: disable=arguments-differ
        if self.use_sigmoid:
            return torch.sigmoid(self.autoencoder(x))
        return self.autoencoder(x)

    def prepare_batch(self, batch):
        '''Assumes batch is a torchio dataset'''
        images = batch['image'][tio.DATA].squeeze(-1)
        masks  = batch['mask'][tio.DATA].squeeze(-1)
        return images, masks

    def training_step(self, batch, *args, **kwargs):
        # pylint: disable=arguments-differ
        images, masks = self.prepare_batch(batch)
        recons = self(images)
        loss = self.loss(recons, masks)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=images.shape[0])
        return loss

    def validation_step(self, batch, *args, **kwargs):
        # pylint: disable=arguments-differ
        images, masks = self.prepare_batch(batch)
        recons = self(images)
        loss = self.loss(recons, masks)
        self.log('val_loss', loss, batch_size=images.shape[0])
        return loss

    def predict_step(self, batch, *args, **kwargs):
        # pylint: disable=arguments-differ, unused-argument
        images, masks = self.prepare_batch(batch)
        return self(images), masks, images

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
