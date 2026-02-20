import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_hmm_incompatible_shapes():
    """
    test various edge cases related to hmm matrices with incompatible shape
      during initialization of the HMM class.
    """

    # test incompatible shapes for provided matrices
    with pytest.raises(ValueError, match="Incompatible shapes for observation_states"):
        hmm = HiddenMarkovModel(
            observation_states=np.array(['obs1', 'obs2']),          # [2]
            hidden_states=np.array(['hidden1', 'hidden2']),
            prior_p=np.array([0.5, 0.5]),
            transition_p=np.array([[0.5, 0.5], [0.5, 0.5]]),
            emission_p=np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]) # [2, 3] (second dim should be 2)
        )
    with pytest.raises(ValueError, match="Incompatible shapes for hidden_states"):
        hmm = HiddenMarkovModel(
            observation_states=np.array(['obs1', 'obs2']),
            hidden_states=np.array(['hidden1', 'hidden2']),             # [2]
            prior_p=np.array([0.5, 0.5]),
            transition_p=np.array([[0.5, 0.5], [0.5, 0.5]]),
            emission_p=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])   # [3, 2] (first dim should be 2)
        )
    with pytest.raises(ValueError, match="Incompatible shapes for hidden_states"):
        hmm = HiddenMarkovModel(
            observation_states=np.array(['obs1', 'obs2']),
            hidden_states=np.array(['hidden1', 'hidden2']),                 # [2]
            prior_p=np.array([0.5, 0.5]),
            transition_p=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),    # [3, 2] (first dim should be 2)
            emission_p=np.array([[0.5, 0.5], [0.5, 0.5]])
        )
    with pytest.raises(ValueError, match="Incompatible shapes for hidden_states"):
        hmm = HiddenMarkovModel(
            observation_states=np.array(['obs1', 'obs2']),
            hidden_states=np.array(['hidden1', 'hidden2']),             # [2]
            prior_p=np.array([0.5, 0.5]),
            transition_p=np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),  # [2, 3] (second dim should be 2)
            emission_p=np.array([[0.5, 0.5], [0.5, 0.5]])
        )


def test_hmm_invalid_probs():
    """
    test various edge cases related to hmm matrices which do not have probabilities
    summing to 1 during initialization of the HMM class.
    """

    # test probability matrices which do not sum to 1
    with pytest.raises(ValueError, match="Prior probabilities do not sum to 1"):
        hmm = HiddenMarkovModel(
            observation_states=np.array(['obs1', 'obs2']),
            hidden_states=np.array(['hidden1', 'hidden2']),
            prior_p=np.array([0.5, 0.6]),                       # sums to 1.1, should raise error
            transition_p=np.array([[0.5, 0.5], [0.5, 0.5]]),
            emission_p=np.array([[0.5, 0.5], [0.5, 0.5]])
        )
    with pytest.raises(ValueError, match="Transition probabilities do not sum to 1 for each row"):
        hmm = HiddenMarkovModel(
            observation_states=np.array(['obs1', 'obs2']),
            hidden_states=np.array(['hidden1', 'hidden2']),
            prior_p=np.array([0.5, 0.5]),
            transition_p=np.array([[0.5, 0.6], [0.5, 0.5]]),    # second column sums to 1.1, should raise error
            emission_p=np.array([[0.5, 0.5], [0.5, 0.5]])
        )
    with pytest.raises(ValueError, match="Emission probabilities do not sum to 1 for each row"):
        hmm = HiddenMarkovModel(
            observation_states=np.array(['obs1', 'obs2']),
            hidden_states=np.array(['hidden1', 'hidden2']),
            prior_p=np.array([0.5, 0.5]),
            transition_p=np.array([[0.5, 0.5], [0.5, 0.5]]),
            emission_p=np.array([[0.5, 0.5], [0.5, 0.6]])       # second column sums to 1.1, should raise error
        )


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

    # run the forward algorithm on the observation sequence and assert correctness
    forward_probability = hmm.forward(mini_input['observation_state_sequence'])
    expected_forward_probability = 0.03506441162109375
    assert np.isclose(forward_probability, expected_forward_probability), f"Expected forward probability {expected_forward_probability}, but got {forward_probability}"

    # run the viterbi algorithm on the observation sequence and assert correctness
    best_hidden_state_sequence = hmm.viterbi(mini_input['observation_state_sequence'])
    expected_hidden_state_sequence = list(mini_input['best_hidden_state_sequence'])
    assert len(best_hidden_state_sequence) == len(expected_hidden_state_sequence), f"Expected best hidden state sequence of length {len(expected_hidden_state_sequence)}, but got sequence of length {len(best_hidden_state_sequence)}"
    assert best_hidden_state_sequence == expected_hidden_state_sequence,f"Expected best hidden state sequence {expected_hidden_state_sequence}, but got {best_hidden_state_sequence}"
    

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

    # run the forward algorithm on the observation sequence and assert correctness
    forward_probability = hmm.forward(full_input['observation_state_sequence'])
    expected_forward_probability = 1.6864513843961343e-11
    assert np.isclose(forward_probability, expected_forward_probability), f"Expected forward probability {expected_forward_probability}, but got {forward_probability}"

    # run the viterbi algorithm on the observation sequence and assert correctness
    best_hidden_state_sequence = hmm.viterbi(full_input['observation_state_sequence'])
    expected_hidden_state_sequence = list(full_input['best_hidden_state_sequence'])
    assert len(best_hidden_state_sequence) == len(expected_hidden_state_sequence), f"Expected best hidden state sequence of length {len(expected_hidden_state_sequence)}, but got sequence of length {len(best_hidden_state_sequence)}"
    assert best_hidden_state_sequence == expected_hidden_state_sequence, f"Expected best hidden state sequence {expected_hidden_state_sequence}, but got {best_hidden_state_sequence}"

