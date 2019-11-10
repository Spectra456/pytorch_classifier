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
from torch.utils.tensorboard import SummaryWriter

def train(device, loader, optimizer,criterion, epoch, writer):
	epoch_loss = 0
	epoch_accuracy = 0
	total = 0
	correct = 0
	model.train()
	progress_bar = tqdm(loader)
	for (data, target) in progress_bar:
		progress_bar.set_description('Train')
		data, target = data.to(device), target.to(device) # On GPU
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		epoch_loss += loss
		total += target.size(0)
		calculate_accuracy(output, target)
		correct += calculate_accuracy(output, target)
	
	epoch_accuracy = correct/total
	epoch_loss = epoch_loss / len(train_loader)

	writer.add_scalar('Loss/train', epoch_loss, epoch)
	writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

	return epoch_loss, epoch_accuracy	


def test(device, loader, optimizer, criterion, epoch, writer):
	with torch.no_grad():
		epoch_loss = 0
		epoch_accuracy = 0
		total = 0
		correct = 0
		model.train()
		progress_bar = tqdm(loader)
		for (data, target) in progress_bar:
			progress_bar.set_description('Test ')
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = criterion(output, target)
			epoch_loss += loss
			total += target.size(0)
			calculate_accuracy(output, target)
			correct += calculate_accuracy(output, target)

		epoch_accuracy = correct/total
		epoch_loss = epoch_loss / len(train_loader)
		writer.add_scalar('Loss/test', epoch_loss, epoch)
		writer.add_scalar('Accuracy/test', epoch_accuracy, epoch)
		
		return epoch_loss, epoch_accuracy	


def calculate_accuracy(output, target):
	preds = torch.max(output.data, 1)[1]
	return (preds == target).sum().item()


size = 128
train_transformations = transforms.Compose([transforms.Resize((size,size)),
											transforms.RandomRotation(15),
											transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),
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
						  batch_size=16,
						  num_workers=8,
						 )



writer = SummaryWriter(flush_secs=13)
device = torch.device('cuda')
model = ImageClassifier(size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
best_test_loss = None

for epoch in range(1, 100):
	print('Epoch {}'.format(epoch))
	train_loss, train_accuracy = train(device, train_loader, optimizer, criterion, epoch, writer)
	test_loss, test_accuracy = test(device, test_loader, optimizer, criterion, epoch, writer)
	best_test_loss = test_loss
	if test_loss < best_test_loss:
		best_test_loss = test_loss
	print('Train loss: {0:.3f}|Train accuracy: {2:.3f}|Test loss: {1:.3f}|Test accuracy: {3:.3f}|'.format(train_loss, test_loss, train_accuracy, test_accuracy))


print('Best loss - {0:.3f}'.format(best_test_loss))


