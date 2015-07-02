#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — June 2015
""" Tools to simulate Markov Chains (MC),
and Hidden Markov Models (HMM)

requires numpy >= 1.7
note: scipy >= 0.16 will be needed to set a non global rvs random seed
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import scipy.stats as stats


class MarkovChain(object):
    def __init__(self, transit, states = None, initial_proba = None):
        '''Markov chain model
        
        Parameters
        ----------
        transit : 2-D array-like
            Transition matrix of the chain
            
            T_ij = Proba(S_{k+1} = j| S_{k} = i)
            
            The sum of each line should be one (stochastic matrix).
        states : sequence, optional
            the name of the states, optional
        initial_proba : 1D array-like
            initial probability of each state
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

    def gen(self, initial_state=None, rng=None):
        '''generator of a sequence of states'''
        gen = self.gen_integers(initial_state, rng)
        
        for s in gen:
            yield self.states[s]
    
    def gen_integers(self, initial_state=None, rng=None):
        '''generator of a sequence of integer states'''
        n = self.n_states
        T = self.transit
        if rng is None:
            rng = np.random.RandomState(seed = None)
        
        # 1) Initial state
        if initial_state is None:
            if self.initial_proba is not None:
                s = rng.choice(n, p=self.initial_proba)
            else:
                s = 0
        else:
            s = initial_state
        
        # 2) Generator (infinite) loop
        while True:
            # 2.1) Yield the current state
            yield s
            # 2.2) Generates the next state :
            proba = T[s, :]
            s = rng.choice(n, p=proba)
        # end while True
    # end gen_integers()


class HiddenMarkov(object):
    def __init__(self, mc, obs_laws):
        '''Hidden Markov model
        
        Parameters
        ----------
        mc : MarkovChain
        obs_laws : sequence or dict
            the observation law for each state.
            The random variable law objects should come from `scipy.stats`.
        '''
        self.mc = mc
        self.obs_laws = obs_laws
    
    def gen(self, initial_state=None, mc_rng=None, obs_seed=None, with_state=False):
        '''generator of a sequence of observations
        
        Observations are generated using the *global* numpy.random state
        (controlled with the seed function), because of scipy <0.16 limitation.
        '''
        np.random.seed(obs_seed)
        mc_gen = self.mc.gen_integers(initial_state, mc_rng)
        
        for s_int in mc_gen:
            s = self.mc.states[s_int]
            law = self.obs_laws[s]
            if with_state:
                yield s_int, law.rvs()
            else:
                yield law.rvs()

    # end gen()

    def state_estim(self, obs):
        '''estimate the state sequence given a series of observations `obs`
        
        using Viterbi algorithm.
        
        Returns
        -------
        s_estim: 1-D array
            the most probable state sequence (as integers)
        '''
        mc = self.mc
        n = mc.n_states
        T = mc.transit
        logT = np.log(T)
        m = len(obs)
        obs_laws = self.obs_laws
        
        
        max_ll = np.zeros((n,m))
        argmax = np.zeros((n,m))
        
        # 1) Init
        initial_proba = mc.initial_proba
        if initial_proba is None:
            initial_proba = np.ones(n)/n
        max_ll[:,0] = np.log(initial_proba) + np.array(
                        [obs_laws[s].logpdf(obs[0]) for s in mc.states])
        argmax[:,0] = 0
        
        for i in range(m-1):
            A = logT.T + max_ll[:,i]
            argmax_next = A.argmax(axis=1)
            argmax[:,i+1] = argmax_next
            max_ll[:,i+1] = A[argmax_next,range(n)] + np.array(
                            [obs_laws[s].logpdf(obs[i+1]) for s in mc.states])
        
        s_seq = np.zeros(m) - 1
        
        s_seq[-1] = np.argmax(max_ll[:,-1])
        for i in range(1,m):
            s_seq[m-i-1] = argmax[s_seq[m-i],m-i]
        return s_seq
    # end state_estim()


if __name__ == '__main__':
    # A quite persistent chain
    T = np.array([
        [0.97, 0.03],
        [0.10, 0.90]
    ])
    mc = MarkovChain(T, ['-', 'P'])
    print(mc.states)
    
    import itertools
    gen = mc.gen()
    
    seq = [str(s) for s in itertools.islice(gen, 30)]
    print(''.join(seq) )
    
    obs_laws = {
        '-':stats.norm(0, .1),
        'P':stats.norm(1, .1),
    }
    hm = HiddenMarkov(mc, obs_laws)
    
    state_obs_seq = [o for o in itertools.islice(hm.gen(with_state=True), 1000)]
    state_obs_seq = np.array(state_obs_seq)
    
    s,o = state_obs_seq.T

    s_seq_estim = hm.state_estim(o)
    
    # Plot
    import matplotlib.pyplot as plt
    plt.plot(o)
    plt.plot(s)
    plt.plot(s_seq_estim)
    
    plt.show()
