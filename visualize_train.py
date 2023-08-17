import pandas as pd
import matplotlib.pyplot as plt

# load the csv file into a pandas DataFrame
df = pd.read_csv('/Users/yuelin_xin/Downloads/kt_log (1).csv', header=None)
df1 = pd.read_csv('/Users/yuelin_xin/Downloads/kt_log (2).csv', header=None)
# df2 = pd.read_csv('/Users/yuelin_xin/Downloads/log (3).csv', header=None)

# give columns appropriate names
df.columns = ['Epoch', 'Accuracy', 'Loss']
df1.columns = ['Epoch', 'Train Acc.', 'Accuracy', 'Loss']
# df2.columns = ['Epoch', 'Accuracy', 'Loss']

# print max accuracy
print(f'Max Accuracy: {df1["Accuracy"].max()}')

plt.figure(figsize=(6, 4))
# plt.plot(df['Epoch'], df['Loss'], label='Loss', color='orange')
# plt.plot(df['Epoch'], df['Accuracy'], label='Val. Accuracy')
plt.plot(df1['Epoch'], df1['Accuracy'], label='Val. Accuracy')
plt.plot(df1['Epoch'], df1['Train Acc.'], label='Train Accuracy')


plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
plt.title('Training Progress', fontsize=10)
plt.show()
