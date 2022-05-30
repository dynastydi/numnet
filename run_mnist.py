from core import numnet as nm
import numpy as np
import csv
import sys

def main():
    
    net = nm.net(784, 256, 10, 0.1)
    try:
        net.load('mnist/h_model.npy', 'mnist/o_model.npy')
        print('saved model found.')
        mnist_test(net)
    except Exception:
        print('loading model unsuccessful.')
        mnist_train(net)
        net.save('mnist/h_model.npy', 'mnist/o_model.npy')
        mnist_test(net)

def mnist_train(net):
    tot = 0
    print('training:')
    for row in csv.reader(open('mnist/mnist_train.csv'), delimiter=','):
        # set target neuron values
        targets = [0.01] * 10
        targets[int(row[0])] = 0.99
        values = np.array(row[1:]).astype(float) / 255.0
        net.train(values, targets)

        tot += 1

        sys.stdout.write(f'\r{"{:,}".format(tot)} / 60,000 images trained.')

def mnist_test(net):
    yay = 0
    tot = 0
    print('testing:')
    for row in csv.reader(open('mnist/mnist_test.csv'), delimiter=','):
        targets = [0.01] * 10
        targets[int(row[0])] = 0.99
        values = np.array(row[1:]).astype(float) / 255.0
        h, out = net.predict(values)
        
        yay += np.argmax(out) == np.argmax(targets)
        tot += 1
        
        sys.stdout.write(f'\r{"{:,}".format(tot)} / 10,000 tests complete, {round((yay / tot * 100), 3)} % accurate.')

    print()

if __name__ == '__main__':
    main()
