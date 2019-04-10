# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import argparse
import torch.optim as optim
import torch.autograd as autograd
from functools import partial
from DataLoader import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import sys
import logging
import os

device = torch.device('cuda:0')
# # parameterization
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
    	p.grad.data.clamp_(-clip_value, clip_value)


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


# training module
def train(epoch, train_iter, model, args, logger):
	if args.use_gpu:
		model.to(device)
		args.pretrained_weights.cuda()

	loss_function = nn.CrossEntropyLoss()
	#optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
	optimizer.zero_grad()
	print("# parameters:", sum(param.numel() for param in model.parameters() if param.requires_grad))
	
	steps = 0
	model.train()
	start = time.time()
	train_loss, train_acc = [], []


	for idx, batch in enumerate(train_iter):
		feature, label  = Variable(batch[0]), Variable(batch[1])
		if args.use_gpu:
			feature, label = feature.cuda(), label.cuda()
		
		prediction = model(feature)
		loss = loss_function(prediction, label)
		correct = (torch.max(prediction, 1)[1].view(label.size()).data == label.data).float().sum()

		acc = correct / len(label.data)
		# train_loss  += loss.item()
		# train_acc += acc.item()
		train_loss.append(loss.item())
		train_acc.append(acc.item())

		loss.backward()
		clip_gradient(model, args.grad_clip)
		optimizer.step()
		steps +=1

		
	
		if steps% 10 == 0:
			print('[epoch/batch]: [%d/%d]; train loss: %.2f; train acc: %.2f'%((epoch +1), (idx +1 ), loss, acc))
			sys.stdout.flush()

	avg_epc_loss, avg_epc_acc = sum(train_loss) /float(len(train_loss)), sum(train_acc) / float(len(train_acc))

	
	end = time.time()
	runtime = end - start
	print('epoch:%d; avg epoch Loss: %.2f; avg epoch acc: %.2f'%((epoch+1), avg_epc_loss, avg_epc_acc), flush=True)
	logger.info('\nepoch: %d, train loss: %.4f, train acc: %.2f, time: %.2f' %(epoch, avg_epc_loss, avg_epc_acc, runtime))
	
	return steps


# evaluation module
def evaluate(val_iter, model, args, logger, mode):
	#avg_epc_loss, avg_epc_acc = 0.0, 0.0
	avg_epc_loss, avg_epc_acc = [], []
	avg_pre, avg_rec, avg_fscore = 0.0,0.0, 0.0
	corrects, size = 0, 0
	batch=0


	predicted = []

	loss_function = nn.CrossEntropyLoss()

	model.eval()
	start = time.time()

	with torch.no_grad():
		for idx, batch in enumerate(val_iter):
			feature, label  = Variable(batch[0]), Variable(batch[1])
			if args.use_gpu:
				feature, label = feature.cuda(), label.cuda()
			prediction = model(feature)
			preds = torch.argmax(prediction, dim=1).cpu().data
			predicted.extend([p.item() for p in preds])
			loss = loss_function(prediction, label)
			corrects += (torch.max(prediction, 1)[1].view(label.size()).data == label.data).sum()
			size +=len(label.data)

			# acc = correct /len(label.data)
			#acc = accuracy_score(label.cpu().data.numpy(), torch.argmax(prediction, dim=1).cpu().data.numpy())
			acc = accuracy_score(label.cpu().data.numpy(), preds.numpy())
			pre = precision_score(label.cpu().data.numpy(), torch.argmax(prediction, dim=1).cpu().data.numpy(), average='macro')
			rec = recall_score(label.cpu().data.numpy(), torch.argmax(prediction, dim=1).cpu().data.numpy(), average='macro')
			fsc = f1_score(label.cpu().data.numpy(), torch.argmax(prediction, dim=1).cpu().data.numpy(), average='macro')
			# pre = precision_score(label.cpu().data.numpy(), preds.numpy(), average='micro')
			# rec = recall_score(label.cpu().data.numpy(), preds.numpy(), average='micro')
			# fsc = f1_score(label.cpu().data.numpy(), preds.numpy(), average='micro')

			#avg_epc_loss += loss.item()
			#avg_epc_acc += acc.item()
			avg_epc_loss.append(loss.item())
			avg_epc_acc.append(acc.item())
			avg_pre += pre
			avg_rec += rec
			avg_fscore += fsc

	#avg_epc_loss, avg_epc_acc = avg_epc_loss / len(val_iter), avg_epc_acc / len(val_iter)
	avg_loss, avg_acc = sum(avg_epc_loss) /float(len(avg_epc_loss)), sum(avg_epc_acc) /float(len(avg_epc_acc))

	avg_pre, avg_rec, avg_fscore = avg_pre / len(val_iter), avg_rec / len(val_iter) , avg_fscore / len(val_iter)

	end = time.time()
	runtime = end - start
	if mode=="development":
		print('dev loss: %.2f; dev acc: %.2f'%(avg_loss, avg_acc))
		logger.info('dev loss: %.4f, dev acc: %.2f, time: %.2f' %(avg_loss, avg_acc, runtime))
	elif mode =="testing":

		print('\nevaluation - loss: {:.6f} precision:{:.4f} recall:{:.4f} f-score:{:.4f} acc: {:.4f}%({}/{}) \n'.format(avg_loss,avg_pre, avg_rec, avg_fscore, avg_acc, corrects, size))
		logger.info('\nevaluation - loss: {:.6f} precision:{:.4f} recall:{:.4f} f-score:{:.4f} acc: {:.4f}%({}/{}) \n'.format(avg_loss,avg_pre, avg_rec, avg_fscore, avg_acc, corrects, size))

	model.train()
	return avg_acc, predicted


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
