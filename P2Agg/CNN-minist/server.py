import argparse
import os
import time

import numpy as np
import torch.fx.traceback
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup
from model.WideResNet import WideResNet


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端的数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=300, help='numer of the clients')
# 随机挑选的客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=0.5,
                    help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=20, help='local train epoch')
# batchsize大小
parser.add_argument('-B', '--batchsize', type=int, default=200, help='local train batch size')  # 10
# 模型名称
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
# parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
# 学习率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset', "--dataset", type=str, default="mnist", help="需要训练的数据集")
# 模型验证频率（通信频率）
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
# n um_comm 表示通信次数，此处设置为1k
parser.add_argument('-ncomm', '--num_comm', type=int, default=50, help='number of communications')  # 1000
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_txt = open("test_accuracy.txt", mode="a")
    test_txt_loss = open("test_loss.txt", mode="a")

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None

    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()

    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    # ResNet网络
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)


    ## 如果有多个GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    net = net.to(dev)

    loss_func = F.cross_entropy

    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}

    for key, var in net.state_dict().items():

        print("张量的维度:" + str(var.shape))
        print("张量的Size" + str(var.size()))
        global_parameters[key] = var.clone()

    time_add_t = []
    time_add_t_sum = 0
    time_start = time.perf_counter()

    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))


        order = np.random.permutation(args['num_of_clients'])

        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]


        sum_parameters = None


        time_add = 0

        for client in tqdm(clients_in_comm):


            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)

            time_mark1 = time.perf_counter()
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
            time_mark2 = time.perf_counter()-time_mark1
            time_add += time_mark2


        time_add_t.append(time_add)

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)


        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        sum_loss = 0

        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)

            loss = loss_func(preds, label)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()


            loss_str = str(loss)
            split_1 = loss_str.split('(', 2)
            print(split_1)
            split_2 = split_1[1].split(',', 2)
            loss_float = float(split_2[0])
            sum_loss += loss_float

            num += 1


        print("\n" + 'accuracy: {}'.format(sum_accu / num))
        print(f'coast:{time.perf_counter() - time_start:.8f}s')
        time_cost = time.perf_counter() - time_start
        print('communicate round' + ' ' + str(i + 1) + ' ' + f'time:{time_cost:.8f}s')

        print("\n" + 'loss: {}'.format(sum_loss / num))
        test_txt_loss.write("communicate round" + ' ' + str(i + 1) + " ")
        test_txt_loss.write('loss:' + ' ' + str(float(sum_loss / num)) + " ")
        test_txt_loss.write("\n")

        test_txt.write("communicate round" + ' ' + str(i + 1) + " ")
        test_txt.write('accuracy:' + ' ' + str(float(sum_accu / num)) + " ")
        test_txt.write('time_cost:' + ' ' + format(time_cost, '.8f') + "\n")
        # t = format(timt_cost, '.8f')
        # test_txt.close()

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

    for j in time_add_t:
        time_add_t_sum += j
    test_txt.write("aggregation time" + ' ' + str(time_add_t_sum) + '\n')
    test_txt.close()
