import string
import numpy as np
import sys

import logging
logging.basicConfig()
log = logging.getLogger('main')
log.setLevel(logging.INFO)

all_letters = string.ascii_lowercase

print(all_letters)


letter_ids = { c : 2 * (i + 1) + 1 for i, c in enumerate(all_letters)}
THRESHOLD = 1/(2 * len(all_letters) + 1)
print(letter_ids)




class PatternMatcher:
    def __init__(self, p):
        self.p   = p
        
        self.pid = [letter_ids[i] for i in self.p]
        self.w = [  2/self.pid[0]  ] + [  1/pi for pi in self.pid[1:]  ]

        self.theta = [2] + [2] * ( len(self.p) - 1)

        log.debug('len(p)    : {}'.format(len(self.p)))
        log.debug('len(pid)  : {}'.format(len(self.pid)))
        log.debug('len(w)    : {}'.format(len(self.w)))
        log.debug('len(theta): {}'.format(len(self.theta)))
        
    def clear_O(self, s):
        self.O = np.zeros([len(s) + 1, len(self.p)], dtype=np.int32)
        log.debug('creating array of shape: {}'.format(self.O.shape))
        
    def g(self, x):
        log.debug('x, abs(x) = {}, {}'.format(x, abs(x)))
        if abs(x) < THRESHOLD:
            return 1

        return 0

    def activation(self, j, text):

        sj = letter_ids[ text[j] ]

        for i in range(len(self.w)):
            wi      = self.w[i]
            Oij     = self.O[j, i - 1]
            theta_i = self.theta[i]

            x = sj * wi + Oij - theta_i
            Oij_new = self.g( x )
            self.O[j + 1, i] = Oij_new

            log.info(
                '  ({}, {}) == ({})sj: {}, wi:{:0.2f}, Oij:{}, theta_i:{} --> x:{:0.2f}, Oij_new: {}'
                .format(j, i, text[j], sj, wi, Oij, theta_i, x, Oij_new))
            
    def match(self, text):
        self.clear_O(text)

        for j in range(len(text)):
            log.info('========= reading {}th element {}'.format(j, text[j]))
            self.activation(j, text)
            print(self.O[:j + 1, :])

        print('{} matches found'.format(self.O[:, -1].sum()))



if __name__ == '__main__':

    if len(sys.argv) > 1:
        p = sys.argv[1]
    else:
        p = input('pattern: ')
        
    matcher = PatternMatcher(p.strip())

    if len(sys.argv) > 2:
        t = sys.argv[2]
        matcher.match(t)

    else:
        while True:
            t = input('text: ')
            matcher.match(t)
