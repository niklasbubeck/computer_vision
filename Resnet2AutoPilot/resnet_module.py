import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl


class ResNet(pl.LightningModule):
    def __init__(self, model_module_params):
        super(ResNet, self).__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=1000, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(in_features=1000, out_features=2, bias=True))
        self.learning_rate = model_module_params["learning_rate"]
        self.mean =  torch.Tensor(model_module_params["mean"])
        self.std =  torch.Tensor(model_module_params["std"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # images = self.normalize_inputs(images)
        outputs = self(self.normalize_inputs(images))
        print(labels)
        loss = F.mse_loss(outputs, labels)
        mae = F.l1_loss(outputs, labels)
        logs={"Loss":{"Train": loss}, "MAE_Train":mae}
        self.log_dict(logs, prog_bar=False, logger=True, on_epoch=True, on_step=False, reduce_fx='mean')
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # images = self.normalize_inputs(images)
        outputs = self(self.normalize_inputs(images))
        loss = F.mse_loss(outputs, labels)
        mae = F.l1_loss(outputs, labels)
        logs={"Loss":{"Val": loss}, "MAE_Val":mae}
        self.log_dict(logs, prog_bar=False, logger=True, on_epoch=True, on_step=False, reduce_fx='mean')
    
    def predict(self, inputs):
        self.eval()
        # inputs = self.normalize_inputs(inputs)
        with torch.no_grad():
            preds = self(inputs)
        return preds
    
    def normalize_inputs(self, inputs):
        self.mean, self.std = self.mean.to(self._device), self.std.to(self.device)
        with torch.no_grad():
            inputs_normalized = (inputs.to(self.device) - self.mean[None,:,None,None])/self.std[None,:,None,None]
        return inputs_normalized

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    pass