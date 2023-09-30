import argparse, json
import datetime
import os
import logging
import torch, random
import time
from loguru import logger

from server import *
from client import *
import models, datasets

if __name__ == '__main__':
    test_txt = open("test_acc.txt", mode="a")

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')

    args = parser.parse_args()


    with open('./conf.json', 'r') as f:  # args.conf
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("\n\n")

    time_start = time.perf_counter()
    for e in range(conf["global_epochs"]):
        print(f"当前是第 {e} 次大循环GlobalEpoch")

        candidates = random.sample(clients, conf["k"])
        print("selected clients are: ")
        for c in candidates:
            print('client_id: ', c.client_id)


        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():

            weight_accumulator[name] = torch.zeros_like(params)


        for c in candidates:
            diff = c.local_train(server.global_model)

            for name, params in server.global_model.state_dict().items():  # ResNet-18有122个层需要更新参数，所以这里执行122次循环(通过调试理解)
                weight_accumulator[name].add_(diff[name])

        time_add_t = []
        time_add_t_sum = 0
        t_GlobalEpoch = server.model_aggregate(weight_accumulator)  # 执行完这行代码后，模型全局参数就更新了
        time_add_t.append(t_GlobalEpoch)

        acc, loss = server.model_eval()
        print(f'coast:{time.perf_counter() - time_start:.8f}s')
        time_cost = time.perf_counter() - time_start
        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
        print("---------------------------------")
        test_txt.write("communicate round" + ' ' + str(e + 1) + " ")
        test_txt.write('accuracy:' + ' ' + str(acc) + " ")
        test_txt.write('loss:' + ' ' + str(loss) + " ")
        test_txt.write('time_cost:' + ' ' + format(time_cost, '.8f') + "\n")
    for j in time_add_t:
        time_add_t_sum += j
    test_txt.write("aggregation time" + ' ' + str(time_add_t_sum) + '\n')

    test_txt.close()