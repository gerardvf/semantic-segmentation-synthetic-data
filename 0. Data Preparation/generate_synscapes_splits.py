import random

ids = list(range(25000))

random.shuffle(ids)
train_ids = ids[:-1000]
val_ids = ids[-1000:-500]
test_ids = ids[-500:]

print(len(train_ids), len(val_ids), len(test_ids))

with open('synscapes_splits.txt', 'w') as f:
    f.write(f'train {" ".join([str(id) for id in train_ids])}\n')
    f.write(f'val {" ".join([str(id) for id in val_ids])}\n')
    f.write(f'test {" ".join([str(id) for id in test_ids])}\n')
