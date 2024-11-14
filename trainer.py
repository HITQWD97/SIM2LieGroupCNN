import torch
from tqdm import tqdm
import os


def Trainer(model, dataloaders, optimizer, scheduler, criterion, device, args):
    f_result = open(os.path.join(args.result, 'result.txt'), 'w+')
    for epoch in tqdm(range(1, args.num_epoch + 1), desc='train'):
        model.train()
        train_loss = 0
        num_total = 0
        acc_total = 0
        for i, minibatch in enumerate(dataloaders['train']):

            img, label = minibatch
            label = label.long()
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            pred_label = model(img)
            loss = criterion(pred_label, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            acc = model(img).max(1)[1].type_as(label).eq(label).cpu().data.numpy().mean()

            acc_total += len(minibatch) * acc
            num_total += len(minibatch)
        scheduler.step()
        if (epoch-1) % 1 == 0 or epoch==100:

            print('[Epoch: %d, Loss: %.4f, Train_Acc: %.3f]' % (
            epoch, (train_loss / len(dataloaders['train'])), (acc_total / num_total)))
            f_result.write('Epoch: %d, Loss: %.4f, Train_Acc: %.3f'%(epoch, (train_loss/len(dataloaders['train'])), (acc_total / num_total))+ '\n')
            if not os.path.exists(os.path.join(args.result, 'checkpoint')):
                os.mkdir(os.path.join(args.result, 'checkpoint'))
            torch.save(model.state_dict(),
                       os.path.join(args.result, 'checkpoint', '%d_model.pth' % epoch))
            torch.save(model.state_dict(), os.path.join(args.result, 'checkpoint', 'end_model.pth'))

        if (epoch-1) % 5 == 0 or epoch==100:
            model.eval()
            num_test = 0
            acc_total_test = 0
            with torch.no_grad():
                for j, minibatch in enumerate(dataloaders['test']):
                    img, label = minibatch
                    img, label = img.to(device), label.to(device)
                    acc = model(img).max(1)[1].type_as(label).eq(label).cpu().data.numpy().mean()
                    acc_total_test += len(minibatch) * acc
                    num_test += len(minibatch)
                print('[Epoch: %d, Test_Acc: %.3f]' % (epoch, (acc_total_test / num_test)))
                f_result.write('Epoch: %d, Test_Acc: %.3f' % (epoch, (acc_total_test / num_test)) + '\n')

    f_result.close()



