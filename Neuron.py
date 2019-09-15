import requests
import math
import os


def getData():
    if not os.path.exists('./data'):
        os.makedirs('./data')
    urls = [
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    ]
    fileNames = [
        'train-labels-idx1-ubyte',
        'train-images-idx3-ubyte',
        't10k-labels-idx1-ubyte',
        't10k-images-idx3-ubyte'
    ]
    for (i, url) in enumerate(urls):
        response = requests.get(url, stream=True)
        totalSize = int(response.headers.get('content-length'))
        blockSize = 1024
        wrote = 0
        print(f"Downloading {fileNames[i]}...")
        with open(f'data/{fileNames[i]}.gz', 'wb') as output:
            for data in response.iter_content(blockSize):
                wrote += output.write(data)
                print(
                    f"\r{wrote}/{math.ceil(totalSize*1000/blockSize)}kB", end="")
        print("\n")


if __name__ == '__main__':
    getData()
