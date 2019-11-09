import torchvision.transforms as transforms
import torch
import sys
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from model import *
import torchvision
from tqdm import tqdm

def train(device, loader, optimizer):
	epoch_loss = 0
	epoch_accuracy = 0
	total = 0
	correct = 0

	model.train()
	
	for (data, target) in tqdm(loader):
		data, target = data.to(device), target.to(device) # On GPU
		optimizer.zero_grad()
		output = model(data)
		loss = F.binary_cross_entropy(output, target)
		loss.backward()
		optimizer.step()
		epoch_loss += loss
		total += target.size(0)
		calculate_accuracy(output, target)
		correct += calculate_accuracy(output, target)
	
	epoch_accuracy = correct/total
	epoch_loss = epoch_loss / len(train_loader)
	
	return epoch_loss, epoch_accuracy	


def test(device, loader, optimizer):
	with torch.no_grad():
		epoch_loss = 0
		epoch_accuracy = 0
		total = 0
		correct = 0
		model.train()

		for (data, target) in tqdm(loader):
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = F.binary_cross_entropy(output, target)
			epoch_loss += loss
			total += target.size(0)
			calculate_accuracy(output, target)
			correct += calculate_accuracy(output, target)

		epoch_accuracy = correct/total
		epoch_loss = epoch_loss / len(train_loader)

		return epoch_loss, epoch_accuracy	


def calculate_accuracy(output, target):
    preds = torch.max(output.data, 1)[1]
    ground_truth = torch.max(target, 1)[1]
    return (preds == ground_truth).sum().item()

size = 64
train_transformations = transforms.Compose([transforms.Resize((size,size)),
											transforms.RandomRotation(15),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transformations = transforms.Compose([transforms.Resize((size,size)),
											transforms.ToTensor(),
											transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = ImageDataset('dataset/train.json', train_transformations)
test_dataset = ImageDataset('dataset/val.json', test_transformations) 

train_loader = DataLoader(train_dataset,
						  batch_size=16,
						  shuffle=True,
						  num_workers=8
						 )

test_loader = DataLoader(test_dataset,
						  batch_size=128,
						  num_workers=8,
						 )




device = torch.device('cuda')
model = ImageClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, 50):
	print('Epoch {}'.format(epoch))
	train_loss, train_accuracy = train(device, train_loader, optimizer)
	test_loss, test_accuracy = test(device, test_loader, optimizer)

	print('Train loss: {0:.3f}| Train accuracy: {2:.3f}| Test loss: {1:.3f}| Test accuracy: {3:.3f}|'.format(train_loss, test_loss*8, train_accuracy, test_accuracy))



