import torch
import torch.nn as nn
import os
import math
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import optim

from model.MLP import TSC_MLP
from model.ResNet import resnet
from data_provider import UCR_data_provider

def work_process(args):
    #数据处理
    UCR_dataset_train = UCR_data_provider(args = args, dataset_type = 'train')
    UCR_dataset_test = UCR_data_provider(args = args, dataset_type = 'test')

    #length使用dataset，dataloader的length被batch_size除过
    #注意drop_last, false则下取整, true上取整
    train_len = (len(UCR_dataset_train) // args.batch_size) * args.batch_size
    test_len = math.ceil(len(UCR_dataset_test) / args.batch_size) * args.batch_size

    #返回一个三维tensor, (batch_size, seq_len, S=1/MS=7)
    train_loader = DataLoader(UCR_dataset_train, batch_size = args.batch_size, shuffle = True, drop_last = True)
    test_loader = DataLoader(UCR_dataset_test, batch_size = args.batch_size, shuffle = False, drop_last = False)

    if args.model == 'MLP':
        model = TSC_MLP(input_size = UCR_dataset_train.input_size, output_size = UCR_dataset_train.output_size)
    elif args.model == 'CNN':#未实现
        model = resnet(input_size = UCR_dataset_train.input_size, output_size = UCR_dataset_train.output_size)
    model.to(args.device)

    loss_fn = nn.CrossEntropyLoss()#多维float32, 一维int64
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    eva_file = args.eva_store_path + '/' + args.model + '/'
    train_epoch_acc = []
    train_epoch_loss = []
    test_epoch_acc = []
    test_epoch_loss = []


    for e in range(args.epochs):
        print("epoch number: {}".format(e+1))
        train_acc = 0
        train_loss = 0
        test_acc = 0
        test_loss = 0

        if args.learning_decay and (e+1)%50 == 0:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate/10)

        #train
        for i, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.float().to(args.device)
            y_train = y_train.to(torch.int64).to(args.device)
            y_pred = model(x_train)
            predicted = torch.max(y_pred.data,1)[1]
            loss = loss_fn(y_pred, y_train)

            train_loss += loss.item()
            train_acc += (predicted == y_train).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_epoch_acc.append(100 * (train_acc.item()/train_len))
        train_epoch_loss.append(train_loss/train_len)
        
        #test

        for i, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.float().to(args.device)
            y_test = y_test.to(torch.int64).to(args.device)
            y_val = model(x_test)
            predicted = torch.max(y_val.data,1)[1]
            loss = loss_fn(y_val, y_test)

            test_loss += loss.item()
            test_acc += (predicted == y_test).sum()
        
        test_epoch_acc.append(100 * (test_acc.item()/test_len))
        test_epoch_loss.append(test_loss/test_len)

    print(train_epoch_acc, train_epoch_loss, test_epoch_acc, test_epoch_loss)
    path_img=eva_file + 'Performance.jpg'
    time = list(range(args.epochs))
    fig,(a1,a2)=plt.subplots(1,2,sharex=False,sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(30)
    plt.subplots_adjust(wspace=0.3,hspace=0)

    x_major_locator=plt.MultipleLocator(5)

    a1.plot(time,train_epoch_acc,color='r',marker='o',label='Train_ACC',linewidth =2.5,markersize = 10)
    a1.plot(time,test_epoch_acc,color='g',marker='v',label='Test_ACC',linewidth =2.5,markersize = 10)
    a1.set_xlabel('Time',fontdict={'size':18},labelpad=-1)
    a1.set_ylabel('ACC:%',fontdict={'size':18})
    a1.tick_params(labelsize=14)
    a1.xaxis.set_major_locator(x_major_locator)
    a1.set_title('Accuarcy',fontsize=20)
    a1.legend(loc=0,prop = {'size':18})

    a2.plot(time,train_epoch_loss,color='r',marker='o',label='Train_Loss',linewidth =2.5,markersize = 10)
    a2.plot(time,test_epoch_loss,color='g',marker='v',label='Test_Loss',linewidth =2.5,markersize = 10)
    a2.set_xlabel('Time',fontdict={'size':18},labelpad=-1)
    a2.set_ylabel('Loss',fontdict={'size':18})
    a2.tick_params(labelsize=14)
    a2.xaxis.set_major_locator(x_major_locator)
    a2.set_title('Loss',fontsize=20)
    a2.legend(loc=0,prop = {'size':18})

    fig.tight_layout()
    plt.savefig(path_img)