import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from arguments import args
from sklearn import metrics
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(2024)
np.random.seed(2024)

if args.server == "grace":
    os.environ['http_proxy'] = '10.73.132.63:8080'
    os.environ['https_proxy'] = '10.73.132.63:8080'
elif args.server == "faster":
    os.environ['http_proxy'] = '10.72.8.25:8080'
    os.environ['https_proxy'] = '10.72.8.25:8080'

class dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, trans=None):
        self.x = inputs
        self.y = targets
        self.trans=trans

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
    root = '/scratch/group/optmai/zhishguo/med/'
    info = INFO[args.data]
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', download=True, root=root)

    test_data = test_dataset.imgs 
    test_labels = test_dataset.labels[:, args.task_index]
    
    test_labels[test_labels != args.pos_class] = 99
    test_labels[test_labels == args.pos_class] = 1
    test_labels[test_labels == 99] = 0

    test_data = test_data/255.0
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels) 

    test_dataset = dataset(test_data, test_labels, trans=eval_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batchsize, shuffle=False, num_workers=0)

    if 1 != args.eval_only:
        train_dataset = DataClass(split='train', download=True, root=root)

        train_data = train_dataset.imgs 
        train_labels = train_dataset.labels[:, args.task_index]
    
        train_labels[train_labels != args.pos_class] = 99
        train_labels[train_labels == args.pos_class] = 1
        train_labels[train_labels == 99] = 0

        train_data = train_data/255.0
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels) 

        train_dataset = dataset(train_data, train_labels, trans=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=0)
 
    from libauc.models import resnet18 as ResNet18
    from libauc.losses import AUCMLoss
    from torch.nn import BCELoss 
    from torch.optim import SGD
    from libauc.optimizers import PESG 
    net = ResNet18(pretrained=False) 
    net = net.cuda()  
    
    if args.loss == "CrossEntropy" or args.loss == "CE" or args.loss == "BCE":
        loss_fn = BCELoss() 
        optimizer = SGD(net.parameters(), lr=0.1)
    elif args.loss == "AUCM":
        loss_fn = AUCMLoss()
        optimizer = PESG(net.parameters(), loss_fn=loss_fn, lr=args.lr, margin=args.margin)
     
    if 1 != args.eval_only:
        train(net, train_loader, test_loader, loss_fn, optimizer, epochs=args.epochs)
    
    # to save a checkpoint in training: torch.save(net.state_dict(), "saved_model/test_model") 
    if 1 == args.eval_only: 
        net.load_state_dict(torch.load(args.saved_model_path)) 
        evaluate(net, test_loader) 

def train(net, train_loader, test_loader, loss_fn, optimizer, epochs):
    for e in range(epochs):
        net.train()
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
            loss = loss_fn(preds, targets) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
