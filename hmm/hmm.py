import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p               # I
        self.transition_p = transition_p    # A
        self.emission_p = emission_p        # E


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """

        # Step 1. Initialize variables
        
       
        # Step 2. Calculate probabilities


        # Step 3. Return final probability 
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        

        # Step 1. Initialize variables
        
        # store probabilities of hidden state at each step (rows are observation states, cols are the hidden states)
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        
        # store best path for traceback
        best_path = np.zeros(len(decode_observation_states))         
        

        # Step 2. Calculate Probabilities
        
        # fill in first row in the viterbi table based on prior probs
        first_obs_idx = self.observation_states_dict[decode_observation_states[0]]
        for hidden_st_idx in range(len(self.hidden_states)):
            viterbi_table[0, hidden_st_idx] = self.prior_p[hidden_st_idx] * self.emission_p[hidden_st_idx, first_obs_idx]

        # fill in the rest of the table based on viterbi algorithm
        for t in range(1, len(decode_observation_states)):

            # compute the index in the emission table of the current observation state
            obs_idx = self.observation_states_dict[decode_observation_states[t]]

            # compute the viterbi probs for each hidden state at time t
            for hidden_st_idx in range(len(self.hidden_states)):

                # initialize the max prob and the best previous state idx which yielded it (for traceback)
                max_prob = 0
                best_prev_st_idx = 0

                # compute the probabilities for each hidden state given each previous hidden states
                for prev_hidden_st_idx in range(len(self.hidden_states)):

                    # compute prob of emission given the current hidden state and the previous hidden state
                    prob = viterbi_table[t-1, prev_hidden_st_idx] * self.transition_p[prev_hidden_st_idx, hidden_st_idx]

                    # if the prob is higher than the current max prob, update the max prob and best_prev_st_idx
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_st_idx = prev_hidden_st_idx

                # update the viterbi table and the best path table
                viterbi_table[t, hidden_st_idx] = max_prob * self.emission_p[hidden_st_idx, obs_idx]
                best_path[t-1] = best_prev_st_idx

        # update the best path for the final time step
        best_path[-1] = np.argmax(viterbi_table[-1, :])


        # Step 3. Traceback 
        
        # convert hidden state indices to names
        best_hidden_state_sequence = [self.hidden_states_dict[idx] for idx in best_path]


        # Step 4. Return best hidden state sequence 
        return best_hidden_state_sequence