import time

import torch

from utils import *
from runx.logx import logx
from os.path import join as ospj


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        batch_mark, batch_text, batch_label = batch
        batch_text = batch_text.to(device)
        # batch_text = batch_text.permute(1, 0)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()

        outputs = model(batch_text)
        _, pred_label = torch.max(outputs, 1)

        loss = criterion(outputs, batch_label)

        acc = label_acc(pred_label.cpu().numpy(), batch_label.cpu().numpy())

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        # print(acc.item())

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_label = torch.tensor([]).to(device)
    epoch_pred = torch.tensor([]).to(device)

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            batch_mark, batch_text, batch_label = batch
            batch_text = batch_text.to(device)
            # batch_text = batch_text.permute(1, 0)
            batch_label = batch_label.to(device)

            outputs = model(batch_text)
            _, pred_label = torch.max(outputs, 1)

            loss = criterion(outputs, batch_label)

            epoch_label = torch.cat((epoch_label, batch_label), 0)
            epoch_pred = torch.cat((epoch_pred, pred_label), 0)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), label_acc(epoch_pred.cpu().numpy(), epoch_label.cpu().numpy())


def test(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_clean_label = torch.tensor([]).to(device)
    epoch_clean_pred = torch.tensor([]).to(device)
    epoch_poisoned_label = torch.tensor([]).to(device)
    epoch_poisoned_pred = torch.tensor([]).to(device)

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            batch_mark, batch_text, batch_label = batch
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)

            outputs = model(batch_text)
            _, pred_label = torch.max(outputs, 1)

            loss = criterion(outputs, batch_label)

            # 区分backdoor和clean
            poisoned_idx = torch.where(batch_mark == 1)
            poisoned_idx = poisoned_idx[0]
            clean_idx = torch.where(batch_mark == 0)
            clean_idx = clean_idx[0]

            batch_poisoned_label = batch_label[poisoned_idx]
            batch_clean_label = batch_label[clean_idx]
            pred_poisoned_label = pred_label[poisoned_idx]
            pred_clean_label = pred_label[clean_idx]

            epoch_clean_label = torch.cat((epoch_clean_label, batch_clean_label), 0)
            epoch_clean_pred = torch.cat((epoch_clean_pred, pred_clean_label), 0)

            epoch_poisoned_label = torch.cat((epoch_poisoned_label, batch_poisoned_label), 0)
            epoch_poisoned_pred = torch.cat((epoch_poisoned_pred, pred_poisoned_label), 0)

            epoch_loss += loss.item()

    clean_acc = label_acc(epoch_clean_pred.cpu().numpy(), epoch_clean_label.cpu().numpy())
    poisoned_acc = label_acc(epoch_poisoned_pred.cpu().numpy(), epoch_poisoned_label.cpu().numpy())

    return epoch_loss / len(iterator), clean_acc, poisoned_acc


def train_model(config, model, optimizer, criterion, train_iterator, valid_iterator, test_iterator):
    model = model.to(config.device)
    criterion = criterion.to(config.device)

    best_valid_loss = float('inf')
    model_name = 'model_p' + str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.pt'
    best_model_path = ospj(config.ckpt_dir, config.mode, 'best_' + model_name)
    # last_model_path = ospj(config.ckpt_dir, config.mode, 'last_' + model_name)
    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Start Train Model [{}]======================'.format(localtime))

    train_acc_cache = {'train': [], 'valid': []}
    acc_name = 'acc_p' + str(int(config.p * 100)) + '_' + str(config.wf) + '_' + config.pos + '.csv'

    for epoch in range(config.epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, config.device)
        # logx.add_scalar('train_loss', train_loss, epoch)
        # logx.add_scalar('train_acc', train_acc, epoch)
        train_acc_cache['train'].append(train_acc)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, config.device)
        # logx.add_scalar('valid_loss', valid_loss, epoch)
        train_acc_cache['valid'].append(valid_acc)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
        # if epoch == config.epochs - 1:
        #     torch.save(model.state_dict(), last_model_path)

        # logx.msg('Epoch: {} | Epoch Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_secs))
        # logx.msg('Train Loss: {} | Train Acc: {}%'.format(train_loss, train_acc * 100))
        # logx.msg('Val. Loss: {} | Val. Acc: {}%'.format(valid_loss, valid_acc * 100))

    save_csv(train_acc_cache, ospj(config.log_dir, config.mode, 'train_' + acc_name))
    # save_csv(valid_acc_cache, ospj(config.log_dir, config.mode, 'valid_' + acc_name))

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_clean_acc, test_poisoned_acc = test(
        model, test_iterator, criterion, config.device
    )
    test_acc_cache = {'clean': [test_clean_acc], 'backdoor': [test_poisoned_acc]}

    # print('Test Loss: {:.3f} | Test Acc: {:.2f}%'.format(test_loss, test_acc * 100))
    logx.msg('Test Loss: {:.3f} | Test Clean Acc: {:.2f}% | Test Poisoned Acc: {:.2f}%'.format(
        test_loss, test_clean_acc * 100, test_poisoned_acc * 100))

    save_csv(test_acc_cache, ospj(config.log_dir, config.mode, 'test_' + acc_name))

    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Finish Train Model [{}]======================'.format(localtime))
    return model
