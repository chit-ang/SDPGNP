import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import argparse
from torch_geometric.data import Data, Dataset
import random
# torch.manual_seed(234)
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(data_list, test_ratio=0.2):
    """
    Args:
        data_list (list): The list of data to be split.
        test_ratio (float): The ratio of data to be used as the test set.

    Returns:
        train_list (list): The training set.
        test_list (list): The test set.
    """
    # 随机打乱列表
    data_list = random.sample(data_list, len(data_list))

    # 计算训练集和测试集的大小
    test_size = int(len(data_list) * test_ratio)

    # 划分数据集
    test_list = data_list[:test_size]
    train_list = data_list[test_size:]

    return train_list, test_list


def neg(nodes, edge_index):
    num_nodes = nodes.shape[0]
    num_edges = edge_index.shape[1]

    edge_set = set(
        [str(edge_index[0, i].item()) + "," + str(edge_index[1, i].item()) for i in range(edge_index.shape[1])])

    redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
    sampled_ind = []
    sampled_edge_set = set([])
    for i in range(5 * num_edges):
        node1 = redandunt_sample[0, i].item()
        node2 = redandunt_sample[1, i].item()
        edge_str = str(node1) + "," + str(node2)
        if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
            sampled_edge_set.add(edge_str)
            sampled_ind.append(i)
        if len(sampled_ind) == num_edges:
            break
    return redandunt_sample[:, sampled_ind]


class LinkPredictor(nn.Module):
    def __init__(self, input_dim):
        super(LinkPredictor, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)

    def reset_parameters(self):
        """重置模型的参数"""
        self.initialize()


# Training loop
def train(graphs, model, optimizer, criterion, device):
    acc = []
    auc = []
    # model.train()
    total_loss = 0
    for graph in graphs:
        node_features = graph['prompt'].view(graph['prompt'].shape[0] * graph['prompt'].shape[1], 1024).to(device)
        # node_features=graph['prompt'].to(device)
        # print(node_features.shape)
        edge_index = graph['edge_index']
        edge_index = edge_index.to(device)
        num_edges = edge_index.size(1)
        negative_samples = neg(node_features, edge_index)
        negative_samples = negative_samples.to(device)
        X_pos = node_features[graph['edge_index'][0, :]] * node_features[graph['edge_index'][1, :]]
        X_neg = node_features[negative_samples[0, :]] * node_features[negative_samples[1, :]]
        y_pos = torch.ones(edge_index.size(1))
        y_neg = torch.zeros(negative_samples.size(1))
        X = torch.cat([X_pos, X_neg], dim=0)
        y = torch.cat([y_pos, y_neg], dim=0)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        # tmp = ((torch.sum(outputs[:num_edges] > 0.5)+torch.sum(outputs[num_edges+1:] < 0.5)).to(torch.float32))/outputs.shape[0]
        outputs = outputs.cpu().detach().numpy()
        true_y = y.cpu().detach().numpy()
        ret_auc = roc_auc_score(true_y, outputs)
        auc.append(ret_auc)
    re = float(sum(auc) / len(auc))
    print(re)
    return re


def test(graphs, model, device):
    acc = []
    num = 0
    s = 0
    model.eval()
    infer_acc = []
    for graph in graphs:
        # node_features=graph['prompt'].to(device)
        node_features = graph['prompt'].view(graph['prompt'].shape[0] * graph['prompt'].shape[1], 1024).to(device)
        edge_index = graph['edge_index']
        edge_index = edge_index.to(device)
        num_negatives = edge_index.shape[1]
        num_edges = edge_index.size(1)
        negative_samples = neg(node_features, edge_index)
        negative_samples = negative_samples.to(device)
        X_pos = node_features[graph['edge_index'][0, :]] * node_features[graph['edge_index'][1, :]]
        X_neg = node_features[negative_samples[0, :]] * node_features[negative_samples[1, :]]
        y_pos = torch.ones(edge_index.size(1))
        y_neg = torch.zeros(negative_samples.size(1))
        X = torch.cat([X_pos, X_neg], dim=0)
        y = torch.cat([y_pos, y_neg], dim=0)
        y = y.to(device)
        outputs = model(X).squeeze()
        # tmp = ((torch.sum(outputs[:num_edges] > 0.5)+torch.sum(outputs[num_edges+1:] < 0.5)).to(torch.float32))/outputs.shape[0]\
        outputs = outputs.cpu().detach().numpy()
        true_y = y.cpu().detach().numpy()
        ret_auc = roc_auc_score(true_y, outputs)
        acc.append(ret_auc)
        # acc.append(tmp)
    re = float(sum(acc) / len(acc))
    return re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.5)
    parser.add_argument('--epoch', type=float, default=50)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graphs_list = torch.load('prompt/prompt_obqa_5_0.1.pt', map_location={'cuda:0': 'cuda:0'})
    print(graphs_list)
    set_seed(42)
    train_data, test_data = split_dataset(graphs_list, 0.5)
    feature_dim = 1024
    hidden_dim = 16
    # model = LinkPredictor(feature_dim)
    model = MLP(feature_dim, hidden_dim).to(device)
    model.to(device)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    acc_test = []
    repeat = 10
    result = []
    for i in range(repeat):
        acc_val = []
        model.initialize()
        # model.reset_parameters()
        for i in range(args.epoch):
            print('epoch:', i)
            model.train()
            acc_val.append(train(train_data, model, optimizer, criterion, device))
            model.eval()
            acc_test.append(test(test_data, model, device))
        print('Final acc:', acc_test[acc_val.index(max(acc_val))])
        result.append(acc_test[acc_val.index(max(acc_val))])

    auc_values = np.array(result)

    # 计算AUC的平均值和标准误差
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)
    # stderr_auc = std_auc / np.sqrt(repeat)
    # means=results.mean(axis=0)
    # std=results.std(axis=0)
    confidence = 1.95 * std_auc / np.sqrt(repeat)
    print(mean_auc)
    print(confidence)





