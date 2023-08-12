import csv

def log_csv(epoch, acc, loss):
    with open('log/kt_log.csv', 'a') as f:
        f.write('{},{},{}'.format(epoch, acc, loss))
