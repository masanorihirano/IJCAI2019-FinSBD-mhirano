#!/usr/bin/env python
# coding: utf-8



def pointWise(WINDOW_SIZE = 10,
              W2V_DIM = 10,
              LANG = "fr",
              HIDDEN_SIZE = 600,
              CUDA_NUM = 0,
              BATCH_SIZE = 1000,
              EPOCHS = 10000
             ):
    import json
    import evaluation_shared_task as evalfun
    import numpy as np
    import os
    import math

    dir_name = "model-"+str(LANG)+"-"+str(WINDOW_SIZE) +"-" + str(W2V_DIM) + "-" + str(HIDDEN_SIZE)
    os.mkdir(dir_name)

    train = json.load(open("data/Train_"+str(LANG)+"_new.json","rb"))
    dev = json.load(open("data/Dev_"+str(LANG)+"_new.json","rb"))

    train["text_splited"] = train["text"].split(" ")
    dev["text_splited"] = dev["text"].split(" ")



    PAD = "<pad>"
    train["sentences_list"] = [PAD for i in range(WINDOW_SIZE)]
    train["sentences_list"].extend([word.lower().replace("\n","") for word in train["text_splited"]])
    train["sentences_list"].extend([PAD for i in range(WINDOW_SIZE)])
    dev["sentences_list"] = [PAD for i in range(WINDOW_SIZE)]
    dev["sentences_list"].extend([word.lower().replace("\n","") for word in dev["text_splited"]])
    dev["sentences_list"].extend([PAD for i in range(WINDOW_SIZE)])

    from gensim.models import word2vec

    model = word2vec.Word2Vec([train["text_splited"]],
                              size=W2V_DIM,
                              min_count=0,
                              window=WINDOW_SIZE,
                              iter=3)
    #モデルの保存
    model.save(dir_name + "/word2vec.gensim.model")

    import nltk
    pos_tag_dict = {}
    for i, (k, v) in enumerate(nltk.data.load('help/tagsets/upenn_tagset.pickle').items()):
        pos_tag_dict[k] = i
    pos_tag_dict["other"] = i + 1

    def w2v(word):
        try:
            word_vector = model.wv[word.lower().replace("\n","")]
        except:
            word_vector = np.array([0.0 for i in range(W2V_DIM)])
        try:
            tag = pos_tag_dict[nltk.pos_tag([word])[0][1]]
        except:
            tag = pos_tag_dict["other"]
        pos_vec = np.array([1.0 if i == tag  else 0.0 for i in range(len(pos_tag_dict))])
        ##数字記号:0, 全部小文字: 1, 全部大文字: 2, 大文字1+小文字: 3, other:4
        class_ul = 0
        if word == PAD:
            class_ul = 4
        elif not word.isalpha():
            class_ul = 0
        elif word.lower() == word:
            class_ul = 1
        elif word.upper() == word:
            class_ul = 2
        elif word[0].upper() == word[0] and word[1:].lower() == word[1:]:
            class_ul = 3
        else:
            class_ul = 4
        capital_vec = np.array([1.0 if i == class_ul else 0.0 for i in range(5)])
        ###"\n"が含まれているか
        if word[-1:] == "\n":
            n_vec = np.array([1.0])
        else:
            n_vec = np.array([0.0])
        return np.concatenate([word_vector, pos_vec, capital_vec,n_vec])

    def mkInput(words_list):
        return np.concatenate([w2v(word) for word in words_list])

    def mkLabel(_data):
        labels = np.array([0 for i in range(len(_data["text_splited"]))])
        for i in _data["begin_sentence"]:
            labels[i] = 1
        for i in _data["end_sentence"]:
            labels[i] = 2
        return labels

    def mkAllInput(_data):
        #labels = mkLabel()
        inputs = mkInput(_data["sentences_list"])
        data_len = int(len(inputs) / len(_data["sentences_list"]))
        return [inputs[i*data_len:(i+WINDOW_SIZE*2+1)*data_len] for i in range(len(_data["text_splited"]))]

    inputs = mkAllInput(train)
    labels = mkLabel(train)


    input_len = len(inputs[0])


    times = int(sum(labels == 0)/sum(labels == 1))

    import torch
    import torchvision
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import numpy as np

    num_classes = 3

    class MLPNet (nn.Module):
        def __init__(self):
            super(MLPNet, self).__init__()
            self.fc1 = nn.Linear(input_len, HIDDEN_SIZE)
            self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
            self.fc3 = nn.Linear(HIDDEN_SIZE, num_classes)
            self.dropout1 = nn.Dropout2d(0.2)
            self.dropout2 = nn.Dropout2d(0.2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            return F.relu(self.fc3(x))

    device = 'cuda:'+str(CUDA_NUM) if torch.cuda.is_available() else 'cpu'
    net = MLPNet().to(device)

    dataset = []
    a = len(inputs[0])
    for i in range(len(inputs)):
        if a != len(inputs[i]):
            print(len(inputs[i]))
        if labels[i] !=0:
            for j in range(times):
                dataset.append((inputs[i],labels[i]))
        else:
            dataset.append((inputs[i],labels[i]))

    from sklearn.model_selection import train_test_split
    train_dataset, test_dataset = train_test_split(dataset,test_size=0.1)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=2, drop_last=True)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    inputs_dev = mkAllInput(dev)

    inputs_dev = torch.tensor(inputs_dev).float().to(device)

    num_epochs = EPOCHS
    digits = int(math.log10(num_epochs))+1

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    avg_val_acc_max = 0

    for epoch in range(num_epochs):
        #エポックごとに初期化
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        #train==============================
        #訓練モードへ切り替え
        net.train()
        #ミニバッチで分割して読み込む
        for i, (_input, _label) in enumerate((train_loader)):
        #for i, (_input, _label) in enumerate((train_loader)):
            #viewで縦横32ピクセルで3チャンネルの画像を１次元に変換
            #toでgpuに転送
            _input, _label = _input.float().to(device), _label.to(device)
            #勾配をリセット
            optimizer.zero_grad()
            #順伝播の計算
            outputs = net(_input)
            #lossの計算
            loss = criterion(outputs, _label)
            #lossのミニバッチ分を溜め込む
            train_loss += loss.item()
            #accuracyをミニバッチ分を溜め込む
            #正解ラベル（labels）と予測値のtop1（outputs.max(1)）が合っている場合に1が返ってきます。
            train_acc += (outputs.max(1)[1] == _label).sum().item()
            #逆伝播の計算
            loss.backward()
            #重みの更新
            optimizer.step()
        #平均lossと平均accuracyを計算
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        #val==============================
        #評価モードへ切り替え
        net.eval()
        #評価するときに必要のない計算が走らないようにtorch.no_gradを使用しています。
        with torch.no_grad():
            for i, (_input, _label) in enumerate(test_loader):        
                _input, _label = _input.float().to(device), _label.to(device)
                outputs = net(_input)
                loss = criterion(outputs, _label)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == _label).sum().item()
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)

        #訓練データのlossと検証データのlossとaccuracyをログで出しています。
        print ('Epoch [{}/{}], Loss: {loss:.10f}, val_loss: {val_loss:.10f}, val_acc: {val_acc:.10f}' 
                       .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
        #最後にグラフをプロットするようにリストに格納
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

        if avg_val_acc > avg_val_acc_max:
            ## save model
            avg_val_acc_max = avg_val_acc
            model_name = dir_name + "/v"+("0"*digits + str(epoch))[-digits:]+".pt"
            torch.save({
                'model': net.state_dict(),
            }, model_name)
            print("Saving the checkpoint...")
        if epoch % 10 == 0:
            result = net(inputs_dev).argmax(1)
            dev2 = {}
            dev2["text"] = dev["text"]
            dev2["begin_sentence"] = []
            dev2["end_sentence"] = []
            for i, _label in enumerate(result):
                if _label == 1:
                    dev2["begin_sentence"].append(i)
                elif _label == 2:
                    dev2["end_sentence"].append(i)
            evalfun.evaluate_result(dev,dev2)

            num_ave = 100
            run_ave = np.array([0.0 for i in range(len(val_acc_list)-num_ave)])
            if len(val_acc_list) >= num_ave:
                for i in range(num_ave):
                    run_ave  += np.array(val_acc_list[i:len(val_acc_list)-num_ave+i])
                run_ave /= num_ave
                max_step = run_ave.argmax()+num_ave
            if len(val_acc_list) >= num_ave:
                print("acc max step:"+str(max_step))
                if len(val_acc_list) > max(max_step * 1.1 , max_step + num_ave * 2):
                    break

if __name__ == '__main__':
    import sys
    args = sys.argv
    
    if len(args) <4:
        print("usage: "+str(args[0])+" window_size w2v_dim lang (HIDDEN_SIZE) (cuda_num) (batch_size) (epochs)")
        sys.exit()

    pointWise(WINDOW_SIZE = int(args[1]),
              W2V_DIM = int(args[2]),
              LANG = str(args[3]),
              HIDDEN_SIZE = int(args[4]) if len(args) >=5 else 600,
              CUDA_NUM = int(args[5]) if len(args) >=6 else 0,
              BATCH_SIZE = int(args[6]) if len(args) >=7 else 1000,
              EPOCHS = int(args[7]) if len(args) >=8 else 10000
             )