import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from arguments import args
from sklearn import metrics
from libauc.models import resnet18, resnet101, resnet152, resnet50, densenet121, densenet169, densenet201, densenet161, resnext101_32x8d, resnext50_32x4d, wide_resnet101_2, wide_resnet50_2
from libauc.losses import AUCMLoss, CompositionalAUCLoss, AveragePrecisionLoss, pAUC_CVaR_Loss, pAUC_DRO_Loss, tpAUC_KL_Loss, CrossEntropyLoss
from torch.optim import SGD
from libauc.optimizers import PESG, PDSCA, SOAP, SOPA, SOPAs, SOTAs
from libauc.sampler import DualSampler
import medmnist
from medmnist import INFO, Evaluator

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2024)
np.random.seed(2024)

idx_loss = [AveragePrecisionLoss, pAUC_CVaR_Loss, pAUC_DRO_Loss, tpAUC_KL_Loss]

neural_network_structures = {
    "resnet18": resnet18(pretrained=False).cuda(),
    "resnet101": resnet101(pretrained=False).cuda(),
    "resnet152": resnet152(pretrained=False).cuda(),
    "resnet50": resnet50(pretrained=False).cuda(),
    "densenet121": densenet121(pretrained=False).cuda(),
    "densenet169": densenet169(pretrained=False).cuda(),
    "densenet201": densenet201(pretrained=False).cuda(),
    "densenet161": densenet161(pretrained=False).cuda(),
    "resnext101_32x8d": resnext101_32x8d(pretrained=False).cuda(),
    "resnext50_32x4d": resnext50_32x4d(pretrained=False).cuda(),
    "wide_resnet101_2": wide_resnet101_2(pretrained=False).cuda(),
    "wide_resnet50_2": wide_resnet50_2(pretrained=False).cuda()
}


class DataSet(torch.utils.data.Dataset):

  def __init__(self, inputs, targets, trans=None):
    self.inputs = inputs
    self.targets = targets
    self.trans = trans

  def __len__(self):
    return self.inputs.size()[0]

  def __getitem__(self, idx):
    if self.trans == None:
      return (self.inputs[idx], self.targets[idx], idx)
    else:
      return (self.trans(self.inputs[idx]), self.targets[idx], idx)
      
def train(net, train_loader, test_loader, loss_fn, optimizer, epochs):
  for e in range(epochs):
    net.train()
    for data, targets, index in train_loader:
      #print("data[0].shape: " + str(data[0].shape))
      #exit()
      targets = targets.to(torch.float32)
      data, targets, index = data.cuda(), targets.cuda(), index.cuda()
      logits = net(data)
      #print("torch.sigmoid(logits):" + str(torch.sigmoid(logits)), flush=True)
      #print("preds:" + str(preds), flush=True)
      #print("targets:" + str(targets), flush=True)
      if loss_fn.__class__ in idx_loss:
        loss = loss_fn(torch.sigmoid(logits), targets, index)
      else:
        loss = loss_fn(torch.flatten(torch.sigmoid(logits)), targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    evaluate(net, test_loader, epoch=e)


def evaluate(net, test_loader, epoch=-1):
  # Testing AUC
  net.eval()
  score_list = list()
  label_list = list()
  for data, targets, _ in test_loader:
    data, targets = data.cuda(), targets.cuda()

    score = net(data).detach().clone().cpu()
    score_list.append(score)
    label_list.append(targets.cpu())
  test_label = torch.cat(label_list)
  test_score = torch.cat(score_list)

  test_auc = metrics.roc_auc_score(test_label, test_score)
  print("Epoch:" + str(epoch) + "Test AUC: " + str(test_auc), flush=True)

def get_data_labels(data, split):
  root = 'data/'
  info = INFO[data]
  DataClass = getattr(medmnist, info['python_class'])
  dataset = DataClass(split=split, download=True, root=root)
  return dataset.labels[:, args.task_index]

def get_data_loader(data, split, batchsize, transform):
  root = 'data/'
  info = INFO[data]
  DataClass = getattr(medmnist, info['python_class'])
  dataset = DataClass(split=split, download=True, root=root)

  data = dataset.imgs
  labels = dataset.labels[:, args.task_index]

  labels[labels != args.pos_class] = 99
  labels[labels == args.pos_class] = 1
  labels[labels == 99] = 0

  data = data / 255.0
  data = torch.tensor(data, dtype=torch.float32)
  labels = torch.tensor(labels)

  dataset = DataSet(data, labels, trans=transform)
  sampler = DualSampler(dataset, batchsize, sampling_rate=0.5)
  if split == 'train':
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        sampler=sampler,
        num_workers=0)
  else:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        num_workers=0)

def get_transform(mode):
  mean = 0.5
  std = 0.5
  if mode == 1:
    mean = 0.75
  elif mode == 2:
    mean = 0.25
  elif mode == 3:
    std = 0.75
  elif mode == 4:
    std == 0.25
  
  return transforms.Compose([
      transforms.ToPILImage(),
      transforms.Grayscale(3),
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
  ])

def eval_model(
  data,
  transform,
  nns,
  test_batchsize,
  model_path,
):
  data_loader = get_data_loader(data, 'test', test_batchsize, get_transform(transform))
  net = neural_network_structures[nns]
  net.load_state_dict(torch.load(model_path))
  evaluate(net, data_loader)

def train_model(
  data,
  transform,
  loss,
  weight_decay,
  nns,
  train_batchsize,
  test_batchsize,
  epochs,
  lr,
  margin,
  model_path,
):
  test_loader = get_data_loader(data, 'test', test_batchsize, get_transform(transform))
  train_loader = get_data_loader(data, 'train', train_batchsize, get_transform(transform))
  train_labels = get_data_labels(data, 'train')
  loss_fns = {
    "AUCM": AUCMLoss(),
    "CAUC": CompositionalAUCLoss(),
    "AP": AveragePrecisionLoss(len(train_labels)),
    "pAUC_CVaR": pAUC_CVaR_Loss(len(train_labels), len([label for label in train_labels if label])),
    "pAUC_DRO": pAUC_DRO_Loss(len(train_labels)),
    "tpAUC_KL": tpAUC_KL_Loss(len(train_labels)),
    "CE": CrossEntropyLoss(),
  }
  net = neural_network_structures[nns]
  loss_fn = loss_fns[loss]

  optimizers = {
      "AUCM": PESG(
        net.parameters(),
        loss_fn=loss_fns["AUCM"],
        lr=lr,
        weight_decay=weight_decay,
        margin=margin
      ),
      "CAUC": PDSCA(
        net.parameters(),
        loss_fn=loss_fns["CAUC"],
        lr=lr,
        weight_decay=weight_decay,
        margin=margin
      ),
      "AP": SOAP(
        net.parameters(),
        loss_fn=loss_fns["AP"],
        lr=lr,
        weight_decay=weight_decay,
        margin=margin
      ),
      "pAUC_CVaR": SOPA(
        net.parameters(),
        loss_fn=loss_fns["pAUC_CVaR"],
        lr=lr,
        weight_decay=weight_decay,
        margin=margin
      ),
      "pAUC_DRO": SOPAs(
        net.parameters(),
        loss_fn=loss_fns["pAUC_DRO"],
        lr=lr,
        weight_decay=weight_decay,
        margin=margin
      ),
      "tpAUC_KL": SOTAs(
        net.parameters(),
        loss_fn=loss_fns["tpAUC_KL"],
        lr=lr,
        weight_decay=weight_decay,
        margin=margin
      ),
      "CE": SGD(
        net.parameters(),
        lr=lr,
        weight_decay=weight_decay,
      ),
  }

  optimizer = optimizers[loss]
  train(net,
        train_loader,
        test_loader,
        loss_fn,
        optimizer,
        epochs=epochs)
  torch.save(net.state_dict(), model_path)

def train_all():
  tr_bs = [8,16,32,64, 128]
  te_bs = [32, 64, 128, 256, 512]
  lr = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
  weight_decay = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
  networks = [nns for nns in neural_network_structures]
  loss = ["CE", "AUCM", "CAUC", "AP", "pAUC_CVaR", "pAUC_DRO", "tpAUC_KL"]
  transform = [i for i in range(5)]
  datas = ["breastmnist", "pneumoniamnist"]

  for data in datas:
    for i in tr_bs:
      model_path = f"saved_model/{data}_trainbatchsize_{i}"
      train_model(data, args.transform, args.loss, args.weight_decay, args.nns, i, args.test_batchsize, args.epochs, args.lr, args.margin, model_path)
    for i in te_bs:
      model_path = f"saved_model/{data}_testbatchsize_{i}"
      train_model(data, args.transform, args.loss, args.weight_decay, args.nns, args.train_batchsize, i, args.epochs, args.lr, args.margin, model_path)
    for i in lr:
      model_path = f"saved_model/{data}_learningrate_{i}"
      train_model(data, args.transform, args.loss, args.weight_decay, args.nns, args.train_batchsize, args.test_batchsize, args.epochs, i, args.margin, model_path)
    for i in networks:
      model_path = f"saved_model/{data}_neuralnetworkstructure_{i}"
      train_model(data, args.transform, args.loss, args.weight_decay, i, args.train_batchsize, args.test_batchsize, args.epochs, args.lr, args.margin, model_path)
    for i in loss:
      model_path = f"saved_model/{data}_lossfunction_{i}"
      train_model(data, args.transform, i, args.weight_decay, args.nns, args.train_batchsize, args.test_batchsize, args.epochs, args.lr, args.margin, model_path)
    for i in weight_decay:
      model_path = f"saved_model/{data}_weightdecay_{i}"
      train_model(data, args.transform, args.loss, i, args.nns, args.train_batchsize, args.test_batchsize, args.epochs, args.lr, args.margin, model_path)
    for i in transform:
      model_path = f"saved_model/{data}_transform_{i}"
      train_model(data, i, args.loss, args.weight_decay, args.nns, args.train_batchsize, args.test_batchsize, args.epochs, args.lr, args.margin, model_path)

def main_worker():
  if args.mode == 0:
    train_model(args.data, args.transform, args.loss, args.weight_decay, args.nns, args.train_batchsize, args.test_batchsize, args.epochs, args.lr, args.margin, args.saved_model_path)
  elif args.mode == 1:
    eval_model(args.data, args.transform, args.nns, args.test_batchsize, args.saved_model_path)
  else:
    train_all()

if __name__ == "__main__":
  main_worker()