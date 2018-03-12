import matplotlib.pyplot as plt
import pdb

def length_hist(name, fname, color):
    f = open(fname)
    lengths = [len(line.split()) for line in f]
    plt.clf()
    plt.title(name)
    plt.xlabel('Words')
    plt.ylabel('Examples')
    n, bins, patches = plt.hist(lengths, 50, facecolor=color)
    plt.savefig(fname + '-scatterplot.png')
    plt.show()


def question_length():
    length_hist('Words in Question', './data/train.question', 'green')


def context_length():
    length_hist('Words in Context', './data/train.context', 'blue')

if __name__ == '__main__':
    context_length()
    # answer_length()
    question_length()
    print('yay')
