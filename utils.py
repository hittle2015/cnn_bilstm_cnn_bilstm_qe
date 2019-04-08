# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import torch.optim as optim
import torch.autograd as autograd
from functools import wraps
from DataLoader import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import sys
import logging
import os
device = torch.device('cuda:0')
# # parameterization

def clip_gradient(optimizer, grad_clip):
	for group in optimizer.param_groups:
		for param in group['params']:	
			if param.grad is not None and param.requires_grad:
				param.grad.data.clamp_(-grad_clip, grad_clip)


def getOptimizer(params,name="adam",lr=1,weight_decay=None, momentum=None,scheduler=None):

	name = name.lower().strip()

	if name=="adadelta":
		optimizer=torch.optim.Adadelta(params, lr=1.0*lr, rho=0.9, eps=1e-06, weight_decay=0).param_groups()
	elif name == "adagrad":
		optimizer=torch.optim.Adagrad(params, lr=1.0*lr, lr_decay=0, weight_decay=0)
	elif name == "sparseadam":
		optimizer=torch.optim.SparseAdam(params, lr=1.0*lr, betas=(0.9, 0.999), eps=1e-08)
	elif name =="adamax":
		optimizer=torch.optim.Adamax(params, lr=2.0*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	elif name =="asgd":
		optimizer=torch.optim.ASGD(params, lr=1.0*lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
	elif name == "lbfgs":
		optimizer=torch.optim.LBFGS(params, lr=1.0*lr, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
	elif name == "rmsprop":
		optimizer=torch.optim.RMSprop(params, lr=1.0*lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
	elif name =="rprop":
		optimizer=torch.optim.Rprop(params, lr=1.0*lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
	elif name =="sgd":
		optimizer=torch.optim.SGD(params, lr=1.0*lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
	elif name =="adam":
		optimizer=torch.optim.Adam(params, lr=0.001*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	else:
		print("undefined optimizer, use adam in default")
		optimizer=torch.optim.Adam(params, lr=0.1*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	
	if scheduler is not None:
		if scheduler == "lambdalr":
			lambda1 = lambda epoch: epoch // 30
			lambda2 = lambda epoch: 0.95 ** epoch
			return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
		elif scheduler=="steplr":
			return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
		elif scheduler =="multisteplr":
			return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
		elif scheduler =="reducelronplateau":
			return  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
		else:
			pass
	else:
		return optimizer
# logging module

def getLogger(fname):
	import random
	random_str = str(random.randint(1,10000))

	now = int(time.time()) 
	timeArray = time.localtime(now)
	timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
	log_filename = "log/" +time.strftime("%Y%m%d", timeArray)

	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program) 
	if not os.path.exists("log"):
		os.mkdir("log")
	if not os.path.exists(log_filename):
		os.mkdir(log_filename)
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',datefmt='%a, %d %b %Y %H:%M:%S',filename=log_filename+'/'+fname+'_'+timeStamp+"_"+ random_str+'.log',filemode='w')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ' '.join(sys.argv))
	return logger

def DataIterator (file_path,seq_len, batch_size):
	data = DataLoader(file_path)
	tokenized_text = data.text
	features = torch.tensor(data._padding(data._encoding(tokenized_text), maxlen=seq_len))
	labels = torch.tensor(data.scores)
	data_set = torch.utils.data.TensorDataset(features, labels)
	data_iter = torch.utils.data.DataLoader(data_set, batch_size=batch_size,shuffle=True)
	return data_iter
#training module
def train (train_iter, dev_iter, test_iter, model, args, logger): #num_epoches, train_file, learning_rate, weight_decay, batch_size
	"""
	parameters:
	vocab: Vocabulary Object
	epoches: number of epoches
	model: initialized neural network
	train_file: path to training file
	learning_rate: configured learning rate
	weight_decay: L2 regularization eg. 1e-6
	"""
	loss_function = nn.CrossEntropyLoss()
	#optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
	optimizer = getOptimizer(model.parameters(), name=args.optimizer, lr = args.learning_rate, momentum=args.momentum, weight_decay= args.weight_decay,scheduler=args.scheduler)

	#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	
	print("# parameters:", sum(param.numel() for param in model.parameters() if param.requires_grad))

	if args.use_gpu is True:
		model.to(device)
		#train_iter, dev_iter, test_iter = train_iter.cuda(), dev_iter.cuda(), test_iter.cuda()
		args.pretrained_weights =args.pretrained_weights.cuda()

	
	best_acc = 0.0
	#model.train()
	best_test_acc = 0.0

	for epoch in range(1,args.num_epochs+1 ):
		model.train()
		start = time.time()
		train_loss, dev_losses = 0, 0
		train_acc, dev_acc = 0, 0


		
		n, m = 0, 0
		for feature, label in train_iter:
			n += 1
			model.zero_grad()
			if args.use_gpu:
				feature, label = Variable(feature.cuda()), Variable(label.cuda())
			score = model(feature)
			loss = loss_function(score, label)
			loss.backward()
			optimizer.step()
			train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
			train_loss += loss
		# tuning on development set
		with torch.no_grad():
			#dev_losses, dev_acc = eval(dev_iter, model, args)
			
			for dev_feature, dev_label in dev_iter:
				m += 1
				dev_feature = dev_feature.cuda()
				dev_label = dev_label.cuda()
				dev_score = model(dev_feature)
				dev_loss = loss_function(dev_score, dev_label)
				dev_acc += accuracy_score(torch.argmax(dev_score.cpu().data,dim=1), dev_label.cpu())


				dev_losses += dev_loss
				if dev_acc > best_acc:
					best_acc = dev_acc
					save(model, args.saved_model, 'best', n)
		#print('n', n)
		#print('m', m)

		train_loss /=n
		train_acc /=n
		dev_acc /= m
		dev_losses /=m

		end = time.time()
		runtime = end - start
		print('epoch: %d, train loss: %.4f, train acc: %.2f, dev loss: %.4f, dev acc: %.2f, time: %.2f' %
			(epoch, train_loss, train_acc, dev_losses, dev_acc, runtime))
		logger.info('epoch: %d, train loss: %.4f, train acc: %.2f, dev loss: %.4f, dev acc: %.2f, time: %.2f' %
			(epoch, train_loss, train_acc, dev_losses, dev_acc, runtime))

		#testing while training----start evaluation
		test_loss, test_acc, predictions = eval(test_iter, model, args, logger)
		# print('test loss: %.4f, test acc: %.2f'%(test_loss, test_acc))
		# logger.info('test loss: %.4f, test acc: %.2f'%(test_loss, test_acc))

		# saving the results with best accuracy on the test data only
		if test_acc > best_test_acc:
			best_test_acc = test_acc
			save_predictions(args.test_file, args.prediction_file, predictions)




def eval(data_iter, model, args, logger):
	model.eval()
	corrects, avg_loss = 0, 0
	precision, recall, fscore = 0.0, 0.0, 0.0
	m, size = 0, 0

	predicted = []

	for feature, label in data_iter:
		m +=1
		if args.use_gpu:
			feature, label = Variable(feature.cuda()), Variable(label.cuda())

		logit = model(feature)
		loss = F.cross_entropy(logit, label, size_average=False)

		avg_loss += loss.item()
		corrects += (torch.max(logit, 1)[1].view(label.size()).data == label.data).sum()
		#print(len(label.data))
		size += len(label.data)
		preds = torch.max(logit, 1)[1]
		predicted.extend(preds.cpu().data.numpy())

		pre = precision_score(label.cpu().data.numpy(), preds.cpu().data.numpy(), average='weighted')
		rec = recall_score(label.cpu().data.numpy(), preds.cpu().data.numpy(), average='weighted')
		fs = f1_score(label.cpu().data.numpy(), preds.cpu().data.numpy(), average='weighted')

		#print(pre, rec, fs)
		precision += pre
		recall += rec
		fscore +=fs
	print('corrects', corrects.item())
	print('size', size)
	avg_loss /= m
	precision /= m
	recall /= m
	fscore /= m
	accuracy = corrects.item() / size
	print('\nEvaluation - loss: {:.6f} Precision:{:.4f} Recall:{:.4f} Fscore:{:.4f} acc: {:.4f}%({}/{}) \n'.format(avg_loss,
		precision,recall, fscore,accuracy, corrects, size))
	logger.info('\nEvaluation - loss: {:.6f} Precision:{:.4f} Recall:{:.4f} Fscore:{:.4f} acc: {:.4f}%({}/{}) \n'.format(avg_loss,
		precision,recall, fscore, accuracy, corrects, size))
	return avg_loss, accuracy, predicted


def save(model, save_dir, save_prefix, steps):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_prefix = os.path.join(save_dir, save_prefix)
	save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
	torch.save(model.state_dict(), save_path)

def save_predictions (test_file, prediction_file, predicted):

	df = pd.read_csv(test_file, delimiter="\t", header=None)
	#print('*'*10)
	#show test data size
	#print(len(predicted))
	# adding an additional column of predictions
	df['predicted']=predicted

	df.to_csv(prediction_file, sep='\t')
