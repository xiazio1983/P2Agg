import models, torch
import time


# 服务器类
class Server(object):

    def __init__(self, conf, eval_dataset):  # 定义构造函数

        self.conf = conf

        self.global_model = models.get_model(self.conf["model_name"])

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)


    def model_aggregate(self, weight_accumulator):

        time_start = time.perf_counter()
        for name, data in self.global_model.state_dict().items():  # ResNet-18有122个层需要更新参数，所以这里执行122次循环(通过调试理解)

            update_per_layer = weight_accumulator[name] * self.conf["lambda"]

            if data.type() != update_per_layer.type():

                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
        time_t = time.perf_counter()-time_start
        return time_t



    def model_eval(self):

        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0

        for batch_id, batch in enumerate(self.eval_loader):  # self.eval_loader有 313 份测试集batch(32张图片为一份batch)
            data, target = batch

            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()


            output = self.global_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss

            pred = output.data.max(1)[1]

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))

        total_l = total_loss / dataset_size

        return acc, total_l