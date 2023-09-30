
import models, torch, copy



class Client(object):  # clients.append(Client(conf, server.global_model, train_datasets, c))

    def __init__(self, conf, model, train_dataset, id=-1):

        self.conf = conf

        self.local_model = models.get_model(self.conf["model_name"])

        self.client_id = id

        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))


    def local_train(self, model):

        for name, param in model.state_dict().items():

            self.local_model.state_dict()[name].copy_(param.clone())


        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])


        self.local_model.train()

        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

                optimizer.step()
            print(f'客户端: {self.client_id} 的', "Epoch %d done." % e)


        diff = dict()

        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])

        print("客户端 %d 本地训练结束" % self.client_id)

        return diff