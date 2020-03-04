import matplotlib.pyplot as plt
import csv

loss_l = []
loss_c = []
loss = []
epoch_loss, epoch_loss_l, epoch_loss_c = 0, 0, 0

with open('logs1.csv') as f:
    reader = csv.reader(f)
    row = next(reader)      # name of column
    for row in reader:
        # iter.append(int(row[1]))
        # loss_l.append(float(row[2]))
        # loss_c.append(float(row[3]))
        # iter_loss.append(float(row[4]))
        if int(row[1]) % 165 != 164:
            epoch_loss += float(row[4])
            epoch_loss_l += float(row[2])
            epoch_loss_c += float(row[3])
        else:
            loss.append(epoch_loss/165)
            loss_l.append(epoch_loss_l/165)
            loss_c.append(epoch_loss_c/165)
            epoch_loss = 0
            epoch_loss_l = 0
            epoch_loss_c = 0

with open('logs.csv') as f:
    reader = csv.reader(f)
    row = next(reader)      # name of column
    for row in reader:
        # iter.append(int(row[1]))
        # loss_l.append(float(row[2]))
        # loss_c.append(float(row[3]))
        # iter_loss.append(float(row[4]))
        if int(row[1]) % 165 != 164:
            epoch_loss += float(row[4])
            epoch_loss_l += float(row[2])
            epoch_loss_c += float(row[3])
        else:
            loss.append(epoch_loss/165)
            loss_l.append(epoch_loss_l/165)
            loss_c.append(epoch_loss_c/165)
            epoch_loss = 0
            epoch_loss_l = 0
            epoch_loss_c = 0

# x = [1, 2, 4]
# y = [4, 8, 16]
# print(loss)
# plt.xlabel('iteration')
# plt.ylabel('epoch_loss')
plt.plot(loss)
plt.show()