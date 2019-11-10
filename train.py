import os
import sys
import torch
import argparse
import datetime
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model import *

def train(device, loader, optimizer,criterion, epoch, writer):
	epoch_loss = 0
	epoch_accuracy = 0
	total = 0 # Num of total ground truth in dataset
	correct = 0 # Correct predictions

	model.train()
	progress_bar = tqdm(loader)

	for (data, target) in progress_bar:
		progress_bar.set_description('Train')
		data, target = data.to(device), target.to(device) 
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

	writer.add_scalar('Loss/train', epoch_loss, epoch) # Tensorboard
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

		writer.add_scalar('Loss/test', epoch_loss, epoch) # Tensorboard
		writer.add_scalar('Accuracy/test', epoch_accuracy, epoch)
		
		return epoch_loss, epoch_accuracy	


def calculate_accuracy(output, target):
	preds = torch.max(output.data, 1)[1] # Get prediction from softmax
	return (preds == target).sum().item() # Compare ground truth with prediction


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=100)  
	parser.add_argument('--batch_size', type=int, default=16) 
	parser.add_argument('--img_size', type=tuple, default=(32, 32), help='input size of image ')
	parser.add_argument('--weights', type=str, default='weights', help='path for saving weight')
	parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='number of threads for data loader, by default using all cores')
	parser.add_argument('--dataset', type=str, default='dataset', help='path of dataset')
	parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
	parser.add_argument('--device', type=str, default='cuda', help='Device for training')

	args = parser.parse_args()

	size = args.img_size
	batch_size = args.batch_size
	num_workers = args.num_workers
	epochs = args.epochs
	path = args.dataset
	learning_rate = args.learning_rate
	device = torch.device(args.device)

	train_transformations = transforms.Compose([transforms.Resize((size[0],size[1])),
												transforms.RandomRotation(15),
												transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),
												transforms.ToTensor(),
												transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	test_transformations = transforms.Compose([transforms.Resize((size[0],size[1])),
												transforms.ToTensor(),
												transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	train_dataset = ImageDataset('{}/train.json'.format(path), train_transformations)
	test_dataset = ImageDataset('{}/val.json'.format(path), test_transformations) 

	train_loader = DataLoader(train_dataset,
							  batch_size=batch_size,
							  shuffle=True,
							  num_workers=num_workers
							 )

	test_loader = DataLoader(test_dataset,
							  batch_size=batch_size,
							  num_workers=num_workers,
							 )



	writer = SummaryWriter(flush_secs=13, log_dir='logs')

	model = ImageClassifier(size).to(device)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()

	best_test_loss = 2
	best_test_accuracy = 0
	MODEL_SAVE_PATH = 'weights'

	if os.path.isdir(MODEL_SAVE_PATH) == False:
		os.mkdir(MODEL_SAVE_PATH)

	for epoch in range(0, epochs):
		print('Epoch {}'.format(epoch))
		train_loss, train_accuracy = train(device, train_loader, optimizer, criterion, epoch, writer)
		test_loss, test_accuracy = test(device, test_loader, optimizer, criterion, epoch, writer)
		
		if test_loss < best_test_loss: # Choosing best epoch and saving
			best_test_loss = test_loss
			best_test_accuracy = test_accuracy
			torch.save(model.state_dict(), MODEL_SAVE_PATH + '/last.pt')

		print('Train loss: {0:.3f}|Train accuracy: {2:.3f}|Test loss: {1:.3f}|Test accuracy: {3:.3f}|'.format(train_loss, test_loss, train_accuracy, test_accuracy))

	torch.save(model.state_dict(), MODEL_SAVE_PATH + '/best_loss{}_accuracy{}.pt'.format(round(best_test_loss.item(), 3), round(best_test_accuracy, 3))) # Saving the best epoch with loss and acc
	print('| Best loss - {0:.3f}| Best accuracy - {1:.3f} |'.format(best_test_loss, best_test_accuracy))
