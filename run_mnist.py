import numnet as nm
import numpy as np
import csv
import sys
import time

def main():
    
    epochs = 5

    net = nm.net(784, 256, 10, 0.1)
    
    # for using previous training:
    # net.load('mnist/h_model.npy', 'mnist/o_model.npy')
    

    for e in range(epochs):
        print(f'epoch {e+1} / {epochs}:')
        #now = time.time()
        mnist_train(net)
        #print(f'in {"{:,}".format(time.time() - now)} seconds.')
        net.save('mnist/h_model.npy', 'mnist/o_model.npy')
        mnist_test(net)

def mnist_train(net):
    tot = 0
    now = time.time()
    for row in csv.reader(open('mnist/mnist_train.csv'), delimiter=','):
        # set target neuron values
        targets = np.array(([0.01] * 10), ndmin=2).astype(float)
        targets[0, int(row[0])] = 0.99
        values = np.array(row[1:], ndmin=2).astype(float) / 255.0

        net.train(values, targets)

        tot += 1

        sys.stdout.write(f'\r{"{:,}".format(tot)} / 60,000 images trained in {round(time.time() - now, 3)} seconds.')
    print()

def mnist_test(net):
    yay = 0
    tot = 0
    for row in csv.reader(open('mnist/mnist_test.csv'), delimiter=','):
        
        targets = np.array(([0.01] * 10), ndmin=2).astype(float)
        targets[0, int(row[0])] = 0.99
        values = np.array(row[1:], ndmin=2).astype(float) / 255.0

        _, out = net.predict(values)
        
        yay += np.argmax(out) == np.argmax(targets)
        tot += 1
        
        sys.stdout.write(f'\r{"{:,}".format(tot)} / 10,000 tests complete, {yay} correct, {round((yay / tot * 100), 3)} % accurate.')

    print()

if __name__ == '__main__':
    main()
