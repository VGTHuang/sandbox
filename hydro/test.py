import numpy as np

def load_csv():
    data = []
    with open(r'C:\Users\hhx\Desktop\srsc.csv', 'r') as f:
        a = f.readline()
        while a:
            a = a[:-2]
            print(a.split(','))
            a = [float(i) for i in a.split(',')]
            print(a)
            if len(a) < 14:
                a.append(0)
            data.append(a)
            a = f.readline()
    data = np.array(data)
    print(data.shape)
    np.save('data', data)

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    # load_csv()
    data = np.load('norm_data.npy')
    print(data.shape)
    for col in range(data.shape[1]):
        col_data = data[:,col]
        cdmin = col_data.min()
        cdmax = col_data.max()
        # print(f'[{cdmax}, {cdmin}],')
        col_data = (col_data - cdmin) / (cdmax - cdmin)

        plt.subplot(data.shape[1], 1, col+1)
    
        plt.plot(col_data, c='r')
        plt.plot(np.log(col_data + 0.1))
        cdmin = np.log(col_data + 0.1).min()
        cdmax = np.log(col_data + 0.1).max()
        print(f'[{cdmax}, {cdmin}],')
    # np.save('norm_data', data)
    plt.show()



   
