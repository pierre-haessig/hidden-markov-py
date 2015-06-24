==========================
Hidden Markov Models tools
==========================

Tools to simulate Markov Chains (MC), and Hidden Markov Models (HMM).

Project address: https://github.com/pierre-haessig/hidden-markov-py

Implemented:

* a `MarkovChain` class, that can generate state sequences,
  with a given transition matrix.
* a `HiddenMarkov` class, that can generate observation sequences,
  with a given underlying Markov chain and given probability distribution
  of the observation, which is state dependent.

Note: there is no restriction about the shape of the probability distribution.
It doesn't have to be Gaussian for example.

TODO:

* Viterbi algorithm, that can estimate the most probable state sequence,
  given a HMM and a sequence of observations.


Usage example
-------------

Imports::

    import numpy as np
    import scipy.stats as stats
    import itertools

Create a Markov chain with two states::

    # A quite persistent chain
    T = np.array([
        [0.97, 0.03],
        [0.10, 0.90]
    ])
    mc = MarkovChain(T, ['-', 'P'])
    print(mc.states)
    
    
    gen = mc.gen()
    
    seq = [str(s) for s in itertools.islice(gen, 30)]
    print(''.join(seq) )

Create a Hidden Markov model derived from the chain:: 

    obs_laws = {
        '-':stats.norm(0, 1),
        'P':stats.norm(1, 1),
    }
    hm = HiddenMarkov(mc, obs_laws)
    
    obs = [o for o in itertools.islice(hm.gen(with_state=True), 1000)]
    
    import matplotlib.pyplot as plt
    plt.plot(obs)
    
    plt.show()


Other Python packages dealing with HMMs:

* `hmmlearn <https://github.com/hmmlearn/hmmlearn>`_ (previously in scikit-learn).
