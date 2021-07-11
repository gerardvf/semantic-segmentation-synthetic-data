import random

notest_ids = list(range(2975))
test_ids = list(range(2975, 3475))

random.shuffle(notest_ids)
train_ids = notest_ids[:-500]
val_ids = notest_ids[-500:]

with open('cityscapes_splits.txt', 'w') as f:
    f.write(f'train {" ".join([str(id) for id in train_ids])}\n')
    f.write(f'val {" ".join([str(id) for id in val_ids])}\n')
    f.write(f'test {" ".join([str(id) for id in test_ids])}\n')
