import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from arguments import args
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2024)
np.random.seed(2024)


class dataset(torch.utils.data.Dataset):

  def __init__(self, inputs, targets, trans=None):
    self.x = inputs
    self.y = targets
    self.trans = trans

  def __len__(self):
    return self.x.size()[0]

  def __getitem__(self, idx):
    if self.trans == None:
      return (self.x[idx], self.y[idx], idx)
    else:
      return (self.trans(self.x[idx]), self.y[idx])


def main_worker():

  train_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Grayscale(3),
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(0.5, 0.5),
  ])

  eval_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Grayscale(3),
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(0.5, 0.5),
  ])

  import medmnist
  from medmnist import INFO, Evaluator
  if not args.train_all:
    root = 'data/'
    info = INFO[args.data]
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', download=True, root=root)

    test_data = test_dataset.imgs
    test_labels = test_dataset.labels[:, args.task_index]

    test_labels[test_labels != args.pos_class] = 99
    test_labels[test_labels == args.pos_class] = 1
    test_labels[test_labels == 99] = 0

    test_data = test_data / 255.0
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels)

    test_dataset = dataset(test_data, test_labels, trans=eval_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batchsize,
                                              shuffle=False,
                                              num_workers=0)

    if not args.eval_only:
      train_dataset = DataClass(split='train', download=True, root=root)

      train_data = train_dataset.imgs
      train_labels = train_dataset.labels[:, args.task_index]

      train_labels[train_labels != args.pos_class] = 99
      train_labels[train_labels == args.pos_class] = 1
      train_labels[train_labels == 99] = 0

      train_data = train_data / 255.0
      train_data = torch.tensor(train_data, dtype=torch.float32)
      train_labels = torch.tensor(train_labels)

      train_dataset = dataset(train_data, train_labels, trans=train_transform)
      train_loader = torch.utils.data.DataLoader(
          train_dataset,
          batch_size=args.train_batchsize,
          shuffle=True,
          num_workers=0)

    from libauc.models import resnet18, resnet101, resnet152, resnet50, densenet121, densenet169, densenet201, densenet161, resnext101_32x8d, resnext50_32x4d, wide_resnet101_2, wide_resnet50_2
    from libauc.losses import AUCMLoss, CompositionalAUCLoss, AveragePrecisionLoss, pAUC_CVaR_Loss, pAUC_DRO_Loss, tpAUC_KL_Loss, PairwiseAUCLoss, meanAveragePrecisionLoss, ListwiseCELoss, NDCGLoss, GCLoss_v1, GCLoss_v2, MIDAM_attention_pooling_loss, MIDAM_softmax_pooling_loss, CrossEntropyLoss, FocalLoss
    from libauc.losses.surrogate import barrier_hinge_loss, hinge_loss, logistic_loss, squared_hinge_loss, squared_loss
    from torch.optim import SGD
    from libauc.optimizers import PESG, PDSCA

    loss_fns = {
        "AUCM": AUCMLoss(),
        "CompositionalAUC": CompositionalAUCLoss(),
        "AveragePrecision": AveragePrecisionLoss(len(train_labels)),
        "pAUC_CVaR": pAUC_CVaR_Loss(len(train_labels), len([label for label in train_labels if label])),
        "pAUC_DRO": pAUC_DRO_Loss(len(train_labels)),
        "tpAUC_KL": tpAUC_KL_Loss(len(train_labels)),
        "PairwiseAUC": PairwiseAUCLoss(),
        "meanAveragePrecision": meanAveragePrecisionLoss(len(train_labels), 2),
        "GC": GCLoss_v1(),
        "GC_v2": GCLoss_v2(),
        "MIDAM_attention_pooling": MIDAM_attention_pooling_loss(len(train_labels)),
        "MIDAM_softmax_pooling": MIDAM_softmax_pooling_loss(len(train_labels)),
        "CE": CrossEntropyLoss(),
        "Focal": FocalLoss(),
    }

    nns = {
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
    net = nns[args.nns]
    loss_fn = loss_fns[args.loss]

    optimizers = {
        "SGD":
        SGD(net.parameters(), lr=args.lr),
        "PESG":
        PESG(net.parameters(),
             loss_fn=loss_fns[args.loss],
             lr=args.lr,
             margin=args.margin),
        "PDSCA":
        PDSCA(net.parameters(), lr=0.1, momentum=0.9)
    }

    optimizer = optimizers[args.optimizer]

    if not args.eval_only:
      train(net,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            epochs=args.epochs)
      torch.save(net.state_dict(), args.saved_model_path)
    else:
      net.load_state_dict(torch.load(args.saved_model_path))
      evaluate(net, test_loader)


def train(net, train_loader, test_loader, loss_fn, optimizer, epochs):
  for e in range(epochs):
    net.train()
    index = 0
    for data, targets in train_loader:
      #print("data[0].shape: " + str(data[0].shape))
      #exit()
      targets = targets.to(torch.float32)
      data, targets = data.cuda(), targets.cuda()
      logits = net(data)
      preds = torch.flatten(torch.sigmoid(logits))
      #print("torch.sigmoid(logits):" + str(torch.sigmoid(logits)), flush=True)
      #print("preds:" + str(preds), flush=True)
      #print("targets:" + str(targets), flush=True)
      if args.loss == 'pAUC_CVaR':
        loss = loss_fn(preds, targets, index)
      else:
        loss = loss_fn(preds, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      index += 1
    evaluate(net, test_loader, epoch=e)


def evaluate(net, test_loader, epoch=-1):
  # Testing AUC
  net.eval()
  score_list = list()
  label_list = list()
  for data, targets in test_loader:
    data, targets = data.cuda(), targets.cuda()

    score = net(data).detach().clone().cpu()
    score_list.append(score)
    label_list.append(targets.cpu())
  test_label = torch.cat(label_list)
  test_score = torch.cat(score_list)

  test_auc = metrics.roc_auc_score(test_label, test_score)
  print("Epoch:" + str(epoch) + "Test AUC: " + str(test_auc), flush=True)


if __name__ == "__main__":
  main_worker()
