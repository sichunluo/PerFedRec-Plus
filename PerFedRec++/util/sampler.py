from random import shuffle,randint,choice,sample
import numpy as np
import sys
import pandas as pd

def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def next_batch_pairwise_fl(data,batch_size,select_user_list, n_negs=1):
    training_data = data.training_data
    df = pd.DataFrame(training_data, columns=['user', 'item', 'rating'])
    all_user_list = data.user.keys()
    u_id_list = select_user_list
    ptr = 0
    data_size = len(training_data)
    for u_id in u_id_list:
        selected_df = df[df['user']==u_id]
        users = selected_df['user'].tolist()
        items = selected_df['item'].tolist()
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def next_batch_pairwise_fl_pse(data,batch_size,select_user_list, n_negs=1):
    training_data = data.training_data
    df = pd.DataFrame(training_data, columns=['user', 'item', 'rating'])
    u_id_list = select_user_list
    ptr = 0
    for u_id in u_id_list:
        selected_df = df[df['user']==u_id]
        users = selected_df['user'].tolist()
        items = selected_df['item'].tolist()
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        len_inter = len(u_idx)
        if len_inter == 1:
            yield u_idx, i_idx, j_idx
        else:
            num_pse = max(1, int(0.1*len_inter))
            random_numbers = sample(range(len_inter), len_inter-num_pse)
            u_idx = u_idx[:(len_inter-num_pse)]
            i_idx = [i_idx[iii] for iii in random_numbers]
            j_idx = [j_idx[iii] for iii in random_numbers]
            for jjj in range(num_pse):
                u_idx.append(u_idx[0])
                i_idx.append(data.item[choice(item_list)])
                j_idx.append(data.item[choice(item_list)])
        yield u_idx, i_idx, j_idx

def next_batch_pairwise_fl_pse2(data,batch_size,select_user_list, n_negs=1):
    training_data = data.training_data
    import pandas as pd
    df = pd.DataFrame(training_data, columns=['user', 'item', 'rating'])
    all_user_list = data.user.keys()
    u_id_list = select_user_list
    ptr = 0
    for u_id in u_id_list:
        selected_df = df[df['user']==u_id]
        users = selected_df['user'].tolist()
        items = selected_df['item'].tolist()
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        u_idx.append(u_idx[0])
        i_idx.append(data.item[choice(item_list)])
        j_idx.append(data.item[choice(item_list)])

        yield u_idx, i_idx, j_idx,[u_idx[0]],[data.item[choice(item_list)]],[data.item[choice(item_list)]]

def next_batch_pointwise(data,batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    ptr = 0
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y


def next_batch_sequence(data, batch_size,n_negs=1,max_len=50):
    training_data = list(data.original_seq.values())
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(range(1,data.item_num+1))
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        y =np.zeros((batch_end-ptr, max_len),dtype=np.int)
        neg = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end =  len(training_data[ptr + n]) > max_len and max_len-1 or len(training_data[ptr + n])-1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
            y[n, :end]=training_data[ptr + n][start+1:]
            negatives=sample(item_list,end)
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) >0:
                negatives = sample(item_list, end)
            neg[n,:end]=negatives
        ptr=batch_end
        yield seq, pos, y, neg, np.array(seq_len,np.int)
