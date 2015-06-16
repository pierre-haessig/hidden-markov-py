#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — June 2015
""" Tools to simulate Markov Chains (MC),
and Hidden Markov Models (HMM)

requires numpy >= 1.7
"""

from __future__ import division, print_function, unicode_literals
import numpy as np


class MarkovChain(object):
    def __init__(self, transit, states = None, initial_proba = None):
        '''Build a Markov chain model
        
        Parameters
        ----------
        
        transit: 2-D array-like
            Transition matrix of the chain
            
            T_ij = Proba(S_{k+1} = j| S_{k} = i)
            
            The sum of each line should be one (stochastic matrix).
        '''
        T = transit
        # T should be a square matrix:
        assert T.ndim == 2 
        assert T.shape[0] == T.shape[1]
        self.transit = T
        
        n_states = T.shape[0]
        self.n_states = n_states
        if states is None:
            states = list(range(n_states))
        else:
            assert len(states) == n_states
        self.states = states
        
        if initial_proba is not None:
            assert len(initial_proba) == n_states
        
        self.initial_proba = initial_proba

    def trajectory_gen(self, initial_state=None, rng=None, integer_states=False):
        '''generator of a sequence of states'''
        n = self.n_states
        T = self.transit
        if integer_states:
            states = np.arange(n)
        else:
            states = self.states
        if rng is None:
            rng = np.random.RandomState(seed = None)
        
        # 1) Initial state
        if initial_state is None:
            if self.initial_proba is not None:
                s = rng.choice(n, p=self.initial_proba)
            else:
                s = 0
        else:
            s = states.index(initial_state)
        
        
        # 2) Generator (infinite) loop
        while True:
            # 2.1) Yield the current state
            yield states[s]
            # 2.2) Generates the next state :
            proba = T[s, :]
            s = rng.choice(n, p=proba)
        # end while True
    # end markov_gen()


if __name__ == '__main__':
    # A quite persistent chain
    T = np.array([
        [0.8, 0.2],
        [0.1, 0.9]
    ])
    mc = MarkovChain(T, ['X', '-'])
    print(mc.states)
    
    import itertools
    gen = mc.trajectory_gen()
    
    seq = [s for s in itertools.islice(gen, 20)]
    print(''.join(seq) )
