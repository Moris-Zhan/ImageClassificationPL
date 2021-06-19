import torch
from LightningFunc.utils import writeCSV, write_Best_model_path
from LightningFunc.optimizer import get_lr

def training_step(self, batch, batch_idx):
    # training_step defined the train loop.
    # It is independent of forward
    x, y = batch
    self.reference_image = x
    out = self.forward(x)
    loss = self.criterion(out, y) 
    
    # acc
    _, predicted = torch.max(out, dim=1)
    train_acc = self.accuracy_score(predicted.cpu(), y.cpu()).clone().detach().requires_grad_(True)

    self.logger.experiment.add_scalars("Loss/Step", {"Train":loss}, self.global_step)

    return {'loss':loss, 'acc':train_acc}

def training_epoch_end(self, outputs): # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_train_acc = torch.stack([x['acc'] for x in outputs]).mean()

    self.logger.experiment.add_scalars("Loss/Epoch", {"Train":avg_loss}, self.current_epoch)

    self.logger.experiment.add_scalars("Accuracy/Epoch", {"Train":avg_train_acc}, self.current_epoch) 

    if(self.current_epoch==1):    
        self.sampleImg=torch.rand((1,3, 512, 512)).cuda()
        self.logger.experiment.add_graph(self.model, self.sampleImg)

    # iterating through all parameters
    for name,params in self.named_parameters():       
        self.logger.experiment.add_histogram(name,params,self.current_epoch)

    # logging reference image       
    # self.logger.experiment.add_image("input",torch.Tensor.cpu(self.reference_image[0][0]),self.current_epoch,dataformats="HW")


    
def validation_step(self, batch, batch_idx):
    x, y = batch
    out = self.forward(x)
    loss = self.criterion(out, y) 

    # loss
    values = {'val_loss':loss}
    self.log_dict(values, logger=True, on_epoch=True)

    # acc
    _, predicted = torch.max(out, dim=1)
    val_acc = self.accuracy_score(predicted.cpu(), y.cpu()).clone().detach().requires_grad_(False)

    return {'val_loss': loss, 'val_acc': val_acc}

def validation_epoch_end(self, outputs): # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.
    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

    self.logger.experiment.add_scalars("Loss/Epoch", {"Val":avg_loss}, self.current_epoch)
    self.logger.experiment.add_scalars("Loss/Step", {"Val":avg_loss}, self.global_step)

    self.logger.experiment.add_scalars("Accuracy/Epoch", {"Val":avg_val_acc}, self.current_epoch) 


    self.write_Best_model_path()

def test_step(self, batch, batch_idx, only_pred = True): #定義 Test 階段
    x, y = batch
    out = self.forward(x)

    # acc
    _, predicted = torch.max(out, dim=1)

    if not only_pred:
        test_acc = self.accuracy_score(predicted.cpu(), y.cpu()).clone().detach().requires_grad_(True)
        return {'test_acc': test_acc, 'predicted': predicted.data[0]}
    else:
        return {'predicted': predicted.data[0]}

def test_epoch_end(self, outputs, only_pred = True): # 在test的一個Epoch結束後，計算平均的Loss及Acc.
    if not only_pred:
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log('avg_val_acc', avg_test_acc)

    total_predict_list = torch.stack([x['predicted'] for x in outputs]).tolist()   

    if len(self.target) != 0:
        self.target['Label'] = total_predict_list        
        tensorboard_logs = self.writeCSV()
        return {'progress_bar': tensorboard_logs}