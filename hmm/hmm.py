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
        
        # store store the forward probabilities of hidden state at each step (rows are observation states, cols are the hidden states)
        forward_table = np.zeros((len(input_observation_states), len(self.hidden_states)))
       

        # Step 2. Calculate probabilities

        # fill in first row in the forward table based on prior probs
        first_obs_idx = self.observation_states_dict[input_observation_states[0]]
        for hidden_st_idx in range(len(self.hidden_states)):
            forward_table[0, hidden_st_idx] = self.prior_p[hidden_st_idx] * self.emission_p[hidden_st_idx, first_obs_idx]

        # fill in the rest of the table based on forward algorithm
        for t in range(1, len(input_observation_states)):

            # compute the index in the emission table of the current observation state
            obs_idx = self.observation_states_dict[input_observation_states[t]]

            # compute the forward probs for each hidden state at time t given the observation state
            for hidden_st_idx in range(len(self.hidden_states)):

                # compute the forward probs for each hidden state given each previous hidden states and the known over observation state
                forward_table[t, hidden_st_idx] = np.sum(forward_table[t-1, :] * self.transition_p[:, hidden_st_idx] * self.emission_p[hidden_st_idx, obs_idx])

        # compute the forward probability of the final observed state by summing over the probs of each hidden state at the final time step
        forward_probability = np.sum(forward_table[-1, :])


        # Step 3. Return final probability        
        return forward_probability


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
        best_path = np.zeros(len(decode_observation_states), dtype=int)

        # store the a backtrace matrix for constructing the best path
        backpointers = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=int)

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

                # compute the probabilities of each hidden state given each previous hidden state
                hidden_probs = viterbi_table[t-1, :] * self.transition_p[:, hidden_st_idx]

                # update the viterbi table using the highest probability and store backpointer
                best_prev_st_idx = int(np.argmax(hidden_probs))
                viterbi_table[t, hidden_st_idx] = hidden_probs[best_prev_st_idx] * self.emission_p[hidden_st_idx, obs_idx]
                backpointers[t, hidden_st_idx] = best_prev_st_idx


        # Step 3. Traceback

        # start from the best final hidden state
        best_path[-1] = np.argmax(viterbi_table[-1, :])

        # traceback through each of the previous steps
        for t in range(len(decode_observation_states)-1, 0, -1):
            best_path[t-1] = backpointers[t, best_path[t]]

        # convert hidden state indices to names
        best_hidden_state_sequence = [self.hidden_states_dict[idx] for idx in best_path]

        # Step 4. Return best hidden state sequence 
        return best_hidden_state_sequence