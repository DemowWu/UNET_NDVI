import logging
import numpy as np
def train(model, train_loader, criterion, epoch, args,optimizer,device):
    model.train()
    num_batches = 0
    avg_loss = []
    avg_loss1 = []
    dt_size = len(train_loader.dataset)
    steps = 0
    # ---------------------------- #
    # gradient accumulate 1
    # ---------------------------- #
    for x, y, _, mask in train_loader:
        inputs = x.to(device)
        labels = y.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        avg_loss.append(loss.item())
        loss = loss / args.accumulation_steps  # 损失标准化
        loss.backward()
        num_batches += 1
        if num_batches % args.accumulation_steps == 0:
            steps += 1
            avg_loss1.append(loss.item())
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 梯度清零

        if num_batches % args.log_step == 0:
            print("%d/%d,loss:%0.6f" % (num_batches, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
            logging.info("%d/%d,loss:%0.6f" % (num_batches, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))

    # avg_loss /= num_batches
    # avg_loss1 /= steps
    print('epoch: {} train_loss1:{} train_loss2:{}'.format(epoch, np.mean(avg_loss), np.mean(avg_loss1)))
    logging.info('epoch: {} train_loss1:{} train_loss2:{}'.format(epoch, np.mean(avg_loss), np.mean(avg_loss1)))

    # ---------------------------- #
    # gradient accumulate 2
    # ---------------------------- #
    # for x, y, _, mask in train_loader:
    #     num_batches += 1
    #     one_batch_loss = 0
    #     optimizer.zero_grad()
    #     for i in range(0, args.batch_size, args.one_batch):
    #         inputs = x[i:i + args.one_batch]
    #         labels = y[i:i + args.one_batch]
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         output = model(inputs)
    #         loss = criterion(output, labels)
    #         loss.backward()
    #         avg_loss += loss.item()
    #         one_batch_loss += loss.item()
    #
    #     optimizer.step()
    #     if num_batches % args.log_step == 0:
    #         print("%d/%d,loss:%0.6f" % (num_batches, (dt_size - 1) // train_loader.batch_size + 1, one_batch_loss))
    #         logging.info("%d/%d,loss:%0.6f" % (num_batches, (dt_size - 1) // train_loader.batch_size + 1, one_batch_loss))


def val(model, val_loader, criterion, epoch,device):
    model.eval()
    num_batches = 0
    losses = []
    # ---------------------------- #
    # gradient accumulate 1
    # ---------------------------- #
    for x, y, _, mask in val_loader:
        inputs = x.to(device)
        labels = y.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        losses.append(loss.item())
        num_batches += 1

    # ---------------------------- #
    # gradient accumulate 2
    # ---------------------------- #
    # for x, y, _, mask in val_loader:
    #     num_batches += 1
    #     for i in range(0, args.batch_size, args.one_batch):
    #         inputs = x[i:i + args.one_batch]
    #         labels = y[i:i + args.one_batch]
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         output = model(inputs)
    #         loss = criterion(output, labels)
    #         avg_loss += loss.item()
    #
    # avg_loss /= num_batches
    print('epoch: ' + str(epoch) + ', val_loss: ' + str(np.mean(losses)))
    logging.info('epoch: ' + str(epoch) + ', val_loss: ' + str(np.mean(losses)))
    return np.mean(losses)