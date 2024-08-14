r'''
https://docs.determined.ai/latest/tutorials/pytorch-mnist-tutorial.html
https://nextjournal.com/gkoehler/pytorch-mnist
https://github.com/PyTorch/examples/blob/master/mnist/main.py
'''
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import argparse


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for trianing (default: 64')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=14, metavar='N',
						help='number of epochs to train (default: 14)')
	parser.add_argument('--lr', type=float, default=1.0, metavar='N',
						help='learing rate (default: 1.0)')
	parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
						help='Learning rate step gamma (default: 0.7)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--dry-run', action='store_true', default=False,
						help='quickly check a single pass')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=True,
						help='For saving the current Model')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	momentum = 0.5

	torch.backends.cudnn.enabled = False
	torch.manual_seed(args.seed)
	device = torch.device('cuda' if use_cuda else 'cpu')

	train_kwargs = {'batch_size': args.batch_size}
	test_kwargs = {'batch_size': args.test_batch_size}
	if use_cuda:
		cuda_kwargs = {'num_workers': 1,
					   'pin_memory': True,
					   'shuffle': True}
		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,))
	])

	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('/files/', train=True, download=True, transform=transform),
		batch_size=args.batch_size, shuffle=True
	)

	test_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('/files/', train=False, download=True, transform=transform),
		batch_size=args.test_batch_size, shuffle=True
	)

	# examples = enumerate(test_loader)
	# batch_idx, (example_data, example_targets) = next(examples) # 迭代器的下一个值
	# print(example_data.shape)
	# show(example_data, example_targets)

	model = Net2().to(device)
	# model.load_state_dict(torch.load('./mnist_cnn.pth'))
	# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum)
	optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
	# optimizer.load_state_dict(torch.load('./mnist_optimizer.pth'))
	scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

	# train_loesses = []
	# train_counter = []
	# test_losses = []
	# test_counter = [i * len(train_loader.dataset) for i in range(args.epochs + 1)]

	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)
		scheduler.step()

	if args.save_model:
		torch.save(model.state_dict(), "mnist_cnn.pth")
		torch.save(optimizer.state_dict(), 'mnist_optimizer.pth')



def show(example_data, example_targets):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	for i in range(6):
		plt.subplot(2,3,i+1)
		plt.tight_layout()
		plt.imshow(example_data[i][0], cmap='gray', interpolation=None)
		plt.title("ground truth: {}".format(example_targets[i]))
		plt.xticks([])
		plt.yticks([])
	plt.show()



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x), 2)))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x)


class Net2(nn.Module):
	def __init__(self):
		super(Net2, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output


def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}]%\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()
			))

			if args.dry_run:
				break

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()#sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)#get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)
	))


if __name__ == '__main__':
	# test()
	main()

