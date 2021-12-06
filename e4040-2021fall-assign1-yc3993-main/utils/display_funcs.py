import numpy as np
import matplotlib.pyplot as plt

def visualize_pics(pics, dims=(28, 28, 1), cmap='gist_stern'):
    ## visualization for the input
    num_pics = pics.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_pics)))
    padding = 2
    figure = np.zeros(((dims[0]+2*padding)*grid_size, (dims[1]+2*padding)*grid_size, dims[2]))
    for r in range(grid_size):
        for c in range(grid_size):
            pid = r*grid_size + c
            if pid < num_pics:
                pic = pics[pid]
                high, low = np.max(pic), np.min(pic)
                pic = 255.0*(pic-low)/(high-low)
                rid = (dims[0]+2*padding)*r
                cid = (dims[1]+2*padding)*c
                figure[rid+padding:rid+padding+dims[0], cid+padding:cid+padding+dims[1], :] = pic

    print('num of feature vectors: {}'.format(num_pics))
    plt.figure(figsize=(10, 10))
    plt.imshow(figure.astype('uint8'), cmap=cmap)
    plt.gca().axis('off')
    plt.show()
