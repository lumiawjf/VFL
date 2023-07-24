import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import time
from Models import Mnist_2NN, Mnist_CNN, FedCNN_BFC, FedADA_LLR
from clients import ClientsGroup, client
from datetime import datetime
import sys
import torch.nn as nn

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=20, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='FedLLR', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.0002, help="learning rate, \
					use value from origin paper as default")
parser.add_argument('-ncomm', '--num_comm', type=int, default=50, help='number of communications')
parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')

# Hang added
parser.add_argument('-nm', '--num_malicious', type=int, default=0, help="number of malicious nodes in the network. malicious node's data sets will be introduced Gaussian noise")
parser.add_argument('-st', '--shard_test_data', type=int, default=0, help='it is easy to see the global models are consistent across clients when the test dataset is NOT sharded')
parser.add_argument('-nv', '--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
# end



if __name__=="__main__":

	date_time = datetime.now().strftime("%m%d%Y_%H%M%S")

	# 1. parse arguments and save to file
	# create folder of logs
	log_files_folder_path = f"VFL/logs/{date_time}"
	os.mkdir(log_files_folder_path)

	# save arguments used 
	args = parser.parse_args()
	args = args.__dict__
	with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
		f.write("Command line arguments used -\n")
		f.write(' '.join(sys.argv[1:]))
		f.write("\n\nAll arguments used -\n")
		for arg_name, arg in args.items():
			f.write(f'\n--{arg_name} {arg}')

	result_list = list(range(args['num_comm']))
	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	net = None
	loss_func = None
	datasetname = None
	if args['model_name'] == 'mnist_2nn':
		net = Mnist_2NN()
		loss_func = F.cross_entropy
		datasetname = 'mnist'
	elif args['model_name'] == 'mnist_cnn':
		net = Mnist_CNN()
		loss_func = F.cross_entropy
		datasetname = 'mnist'
	elif args['model_name'] == 'FedBFC':
		net = FedCNN_BFC()
		loss_func = nn.BCELoss()
		datasetname = 'UJIBFC'
	elif args['model_name'] == 'FedLLR':
		net = FedADA_LLR()
		loss_func = nn.L1Loss()
		datasetname = 'UJILLR'
	net = net.to(dev)

	# opti = optim.SGD(net.parameters(), lr=args['learning_rate'])
	myClients = ClientsGroup(datasetname, args['IID'], args['num_of_clients'], args['learning_rate'], dev, net, args['num_malicious'], args['noise_variance'], shard_test_data=args['shard_test_data'])
	# testDataLoader = myClients.test_data_loader

	num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
	global_parameters = net.state_dict()
	for i in range(args['num_comm']):
		comm_round_start_time = time.time()
		print("communicate round {}".format(i+1))

		comm_round_folder = f"{log_files_folder_path}/comm_{i+1}"
		os.mkdir(comm_round_folder)

		order = np.random.permutation(args['num_of_clients'])
		clients_in_comm = ['client_{}'.format(i+1) for i in order[0:num_in_comm]]

		sum_parameters = None
		for client in clients_in_comm:
			myClients.clients_set[client].reset_variance_of_noise()
			local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], loss_func, global_parameters, comm_round_folder, i)
			if sum_parameters is None:
				sum_parameters = local_parameters
			else:
				for var in sum_parameters:
					sum_parameters[var] = sum_parameters[var] + local_parameters[var]

		for var in global_parameters:
			global_parameters[var] = (sum_parameters[var] / num_in_comm)

		clients_list = list(myClients.clients_set.values())
		print(''' Logging Accuracies by clients ''')
		# open(f"{log_files_folder_path}/comm_{i+1}.txt", 'w').close()

		for client in clients_list:
			accuracy_this_round = client.evaluate_model_weights(global_parameters)
			result_list[i] = accuracy_this_round.item()
			with open(f"{comm_round_folder}/global_comm_{i+1}.txt", "a") as file:
				is_malicious_node = "M" if client.is_malicious else "B"
				file.write(f"{client.idx} {is_malicious_node}: {accuracy_this_round}\n")

		# logging time
		comm_round_spent_time = time.time() - comm_round_start_time
		with open(f"{comm_round_folder}/global_comm_{i+1}.txt", "a") as file:
			file.write(f"comm_round_block_gen_time: {comm_round_spent_time}\n")

	print(result_list)
