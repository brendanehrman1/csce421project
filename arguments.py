import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--loss', type=str, default="AUCM")
parser.add_argument('--train_batchsize', type=int, default=32)
parser.add_argument('--epochs', type=int, default=1) 
parser.add_argument('--lr', type=float, default=0.1) 
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--eval_only', type=int, default=0)
parser.add_argument('--saved_model_path', type=str, default="saved_model/test_model")
parser.add_argument('--test_batchsize', type=int, default=128)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--data', type=str, default="breastmnist") 
parser.add_argument('--server', type=str, default="faster")
parser.add_argument('--pos_class', type=int, default=0)
parser.add_argument('--task_index', type=int, default=0)


args = parser.parse_args()
