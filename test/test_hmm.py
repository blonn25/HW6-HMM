import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    # load in the data for the mini weather HMM
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # initialize the HMM object
    hmm = HiddenMarkovModel(
        observation_states=mini_hmm['observation_states'],
        hidden_states=mini_hmm['hidden_states'],
        prior_p=mini_hmm['prior_p'],
        transition_p=mini_hmm['transition_p'],
        emission_p=mini_hmm['emission_p']
    )

    # run the forward algorithm on the observation sequence
    forward_probability = hmm.forward(mini_input['observation_state_sequence'])

    # run the viterbi algorithm on the observation sequence
    my_best_hidden_state_sequence = hmm.viterbi(mini_input['observation_state_sequence'])

    # check that the forward probability is correct
    # assert np.isclose(forward_probability, mini_input['forward_probability']), f"Expected forward probability {mini_input['forward_probability']}, but got {forward_probability}"

    # check that the output of my viterbi algorithm is correct
    assert my_best_hidden_state_sequence == list(mini_input['best_hidden_state_sequence']),f"Expected best hidden state sequence {mini_input['best_hidden_state_sequence']}, but got {my_best_hidden_state_sequence}"


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    # load in the data for the full weather HMM
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    # initialize the HMM object
    hmm = HiddenMarkovModel(
        observation_states=full_hmm['observation_states'],
        hidden_states=full_hmm['hidden_states'],
        prior_p=full_hmm['prior_p'],
        transition_p=full_hmm['transition_p'],
        emission_p=full_hmm['emission_p']
    )

    # run the forward algorithm on the observation sequence
    forward_probability = hmm.forward(full_input['observation_state_sequence'])

    # run the viterbi algorithm on the observation sequence
    my_best_hidden_state_sequence = hmm.viterbi(full_input['observation_state_sequence'])

    # check that the forward probability is correct
    # assert np.isclose(forward_probability, full_input['forward_probability']), f"Expected forward probability {full_input['forward_probability']}, but got {forward_probability}"

    # check that the output of my viterbi algorithm is correct
    assert my_best_hidden_state_sequence == list(full_input['best_hidden_state_sequence']),f"Expected best hidden state sequence {full_input['best_hidden_state_sequence']}, but got {my_best_hidden_state_sequence}"



# mini weather:

# ---- sequence ----

# observation_state_sequence
# ['sunny' 'rainy' 'rainy' 'sunny' 'rainy']

# best_hidden_state_sequence
# ['hot' 'cold' 'cold' 'hot' 'cold']


# ---- hmm ----

# observation_states
# ['sunny' 'rainy']

# prior_p
# [0.6 0.4]

# transition_p
# [[0.55 0.45]
#  [0.3  0.7 ]]

# emission_p
# [[0.65 0.35]
#  [0.15 0.85]]









