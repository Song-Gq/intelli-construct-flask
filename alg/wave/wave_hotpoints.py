# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # load the sample data
# df = pd.DataFrame({'MutProb': [0.1,
#   0.05, 0.01, 0.005, 0.001, 0.1, 0.05, 0.01, 0.005, 0.001, 0.1, 0.05, 0.01, 0.005, 0.001, 0.1, 0.05, 0.01, 0.005, 0.001, 0.1, 0.05, 0.01, 0.005, 0.001],
#                    'SymmetricDivision': [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2],
#                    'test': ['sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule', 'sackin_yule'],
#                    'value': [-4.1808639999999997, -9.1753490000000006, -11.408113999999999, -10.50245, -8.0274750000000008, -0.72260200000000008, -6.9963940000000004, -10.536339999999999, -9.5440649999999998, -7.1964070000000007, -0.39225599999999999, -6.6216390000000001, -9.5518009999999993, -9.2924690000000005, -6.7605589999999998, -0.65214700000000003, -6.8852289999999989, -9.4557760000000002, -8.9364629999999998, -6.4736289999999999, -0.96481800000000006, -6.051482, -9.7846860000000007, -8.5710630000000005, -6.1461209999999999]})
#
# # pivot the dataframe from long to wide form
# result = df.pivot(index='SymmetricDivision', columns='MutProb', values='value')
#
# sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
# plt.show()


# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
#
# df = pd.read_excel("C:/Users/17389/Desktop/wave/test1.xlsx")
# #sns.set(font_scale=1.25)#字符大小设定
# df = abs(df)
# print(df)
# print("df的类型", type(df))
# hm=sns.heatmap(df,vmin=-1,vmax=1,cmap="RdBu_r")
# plt.show()


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# data = np.array([[1,2,3],[4,5,6],[7,8,9]])
# sns.heatmap(data,annot=True)
# plt.show()

# n*30*200
A = []
file = open('C:\\Users\\17389\\Desktop\\wave\\CH04.txt','r')
try:
    for line in file:
        A.append(line)
finally:
    file.close()
num_A = int(len(A)/6000)

B = [[[0 for i in range(200)] for j in range(30)] for k in range(num_A)]
for i in range(num_A):
    for j in range(30):
        for k in range(200):
            B[i][j][k] = abs(float(A[i*6000 + j*200 + k]))

B = np.array(B)

fig_name = '1920_1950s_1.png'
filepath = 'results'
fig_path = filepath + '/' + fig_name
fig = sns.heatmap(B[63],vmin=0,vmax=2,cmap="RdBu_r")
heatmap = fig.get_figure()
heatmap.savefig(fig_path, dpi = 400)


# plt.show()