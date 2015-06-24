==========================
Hidden Markov Models tools
==========================

Tools to simulate Markov Chains (MC), and Hidden Markov Models (HMM).

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
  

Other Python packages dealing with HMMs:

* `hmmlearn <https://github.com/hmmlearn/hmmlearn>`_ (previously in scikit-learn).
