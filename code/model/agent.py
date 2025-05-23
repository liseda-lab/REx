import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

class Agent(object):

    def __init__(self, params):
        self.prevent_cycles =params['prevent_cycles'] #to avoid cycles in the graph
        self.guiding_ic=params['agent_IC_guiding']
        self.weighted_reward=params['weighted_reward']
        self.adjust_factor=params['IC_importance']
        self.sigmoid=params['sigmoid']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.entity_initializer = tf.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        with tf.variable_scope("action_lookup_table"):
            self.action_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            self.relation_lookup_table = tf.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.variable_scope("entity_lookup_table"):
            self.entity_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        with tf.variable_scope("policy_step"):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.contrib.rnn.LSTMCell(self.m * self.hidden_size, use_peepholes=True, state_is_tuple=True))
            self.policy_step = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def policy_MLP(self, state):
        with tf.variable_scope("MLP_for_policy"):
            hidden = tf.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.layers.dense(hidden, self.m * self.embedding_size, activation=tf.nn.relu)
        return output

    def action_encoder(self, next_relations, next_entities):
        with tf.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    def step(self, next_relations, next_entities, next_weights, prev_state, prev_relation, query_embedding, current_entities,
             label_action, range_arr, first_step_of_test):
        """
        Executes one step of the policy, updating state and scoring actions.

        Args:
            next_relations: Candidate relations for the next step [B, MAX_NUM_ACTIONS].
            next_entities: Candidate entities for the next step [B, MAX_NUM_ACTIONS].
            next_weights: Edge weights corresponding to candidate actions [B, MAX_NUM_ACTIONS].
            prev_state: Previous LSTM state.
            prev_relation: Previous relation taken by the agent.
            query_embedding: Embedding of the query relation [B, 2D].
            current_entities: Current entities [B].
            label_action: Ground truth action for supervised training [B].
            range_arr: Range array to map indices.
            first_step_of_test: Flag indicating if it's the first step during testing.

        Returns:
            loss: Sparse softmax cross-entropy loss for the selected action.
            new_state: Updated LSTM state after the step.
            logits: Logits (non-normalized scores) for the candidate actions [B, MAX_NUM_ACTIONS].
            action_idx: Index of the chosen action.
            chosen_relation: Relation corresponding to the chosen action.
        """

        # Compute embeddings for the previous action
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)

        # Performe one step of the LSTM with the previous action embedding
        output, new_state = self.policy_step(prev_action_embedding, prev_state)  # output: [B, 4D]

        # Compute the current state vector
        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        if self.use_entity_embeddings:
            state = tf.concat([output, prev_entity], axis=-1) # include the entity embeddings
        else:
            state = output
        
        # Encode the candidate actions using relation and entity embeddings
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        state_query_concat = tf.concat([state, query_embedding], axis=-1) # Concatenate the state and query embeddings

        # Apply MLP to compute scores for the candidate actions
        output = self.policy_MLP(state_query_concat)
        output_expanded = tf.expand_dims(output, axis=1)  # Expand dimensions to match candidate embeddings 

        # Compute preliminary scores by matching actions with the policy output (logits)
        # logits are the dot product between the agent state and the embeddings of the candidate actions
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, output_expanded), axis=2)
    
        # Apply edge weights to the preliminary scores to prioritize certain actions
        if self.guiding_ic:
            prelim_scores = prelim_scores * next_weights  # MULTIPLY BY WEIGHTS OF THE EDGES

        # Masking PAD actions by setting their scores to a very low value to avoid selection
        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # Tensor for comparison
        mask = tf.equal(next_relations, comparison_tensor)  # Mask for PAD actions
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # Scores for PAD actions
        scores = tf.where(mask, dummy_scores, prelim_scores)  # Final scores with PAD actions masked

        # Sample action from the scores 
        #apply softmax to the scores to get the probabilities of the actions
        action = tf.to_int32(tf.multinomial(logits=scores, num_samples=1))  # [B, 1]

        # Compute loss for the selected action
        label_action =  tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]

        # Map back to the true action indices
        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))

        return loss, new_state, tf.nn.log_softmax(scores), action_idx, chosen_relation
    

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities, next_weights_sequence, 
                 path_label, query_relation, range_arr, first_step_of_test, T=3, entity_sequence=0):

        self.baseline_inputs = []
        # get the query vector
        query_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)  # [B, 2D]
        state = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        prev_relation = self.dummy_start_label

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []

        with tf.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_possible_relations = candidate_relation_sequence[t]  # [B, MAX_NUM_ACTIONS, MAX_EDGE_LENGTH]
                next_possible_entities = candidate_entity_sequence[t]
                current_entities_t = current_entities[t]

                path_label_t = path_label[t]  # [B]
                
                next_weights = next_weights_sequence[t] 


                loss, state, logits, idx, chosen_relation = self.step(next_possible_relations,
                                                                              next_possible_entities,
                                                                              next_weights, # EDGE WEIGHTS
                                                                              state, prev_relation, query_embedding,
                                                                              current_entities_t,
                                                                              label_action=path_label_t,
                                                                              range_arr=range_arr,
                                                                              first_step_of_test=first_step_of_test, 
                                                                              )

                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(idx)
                prev_relation = chosen_relation

            # [(B, T), 4D]

        return all_loss, all_logits, action_idx
