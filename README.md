![BuildStatus](https://github.com/blonn25/HW6-HMM/actions/workflows/test.yml/badge.svg?event=push)

# HW6-HMM

In this assignment, I implemented the Forward and Viterbi Algorithms - dynamic programming approaches for determining forward probability of a given observation sequence and the most probable hidden state sequence for a given observation sequence, respectively. These algorithms are implemented as part of the `HiddenMarkovModel` class in the `forward()` and `viterbi()` methods.

1. Forward Algorithm (`HiddenMarkovModel.forward()`): takes a given observation sequence and computes the forward probability of that sequence given the hmm model. This is done by completing a dynamic programming matrix where each entry the sum of all possible hidden state sequences that could have generated the observation sequence up to that point. The final forward probability is calculated by summing the probabilities of the final columns of the dynamic programming matrix and returned to the user.

2. Viterbi Algorithm (`HiddenMarkovModel.viterbi()`): takes a given observation sequence and computes the most probable hidden state sequence assocaited with that observation sequence given the hmm model. This is accomplished be completing a dynamic programming matrix where each entry is the probability of the most probable path to that point, and a backtracing matrix is stored to enable reconstruction of the most probable path upon completion of the matrix. The most probable path is then returned to the user as a list of hidden states.


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

For a helpful refresher on HMMs and the Forward and Viterbi Algorithms you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 





## Tasks and Data 
Please complete the `forward` and `viterbi` functions in the HiddenMarkovModel class. 

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Start with the mini_weather_hmm model for testing and debugging. Both include the following arrays:
* `hidden_states`: list of possible hidden states 
* `observation_states`: list of possible observation states 
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`) 
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)



For both datasets, we also provide input observation sequences and the solution for their best hidden state sequences. 
 * `observation_state_sequence`: observation sequence to test 
* `best_hidden_state_sequence`: correct viterbi hidden state sequence 


Create an HMM class instance for both models and test that your Forward and Viterbi implementation returns the correct probabilities and hidden state sequence for each of the observation sequences.

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions? 
  * Are the inputs in compatible shapes/sizes which each other? 
  * Any other edge cases you can think of?
  * Ensure that your code accomodates at least 2 possible edge cases. 

Finally, please update your README with a brief description of your methods. 



## Task List

[TODO] Complete the HiddenMarkovModel Class methods  <br>
  [ ] complete the `forward` function in the HiddenMarkovModelClass <br>
  [ ] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[TODO] Unit Testing  <br>
  [ ] Ensure functionality on mini and full weather dataset <br>
  [ ] Account for edge cases 

[TODO] Packaging <br>
  [ ] Update README with description of your methods <br>
  [ ] pip installable module (optional)<br>
  [ ] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](https://forms.gle/xw98ZVQjaJvZaAzSA)

### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)
