import os

def log_csv(epoch, train_acc, val_acc, loss):
    # check if directory exists
    if not os.path.exists('log'):
        os.makedirs('log')
    with open('log/kt_log.csv', 'a') as f:
        f.write('{},{},{},{}\n'.format(epoch, train_acc, val_acc, loss))
