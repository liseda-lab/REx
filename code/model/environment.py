from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging
logger = logging.getLogger()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Episode(object):

    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher, weighted_reward, adjust_factor, sigmoid, size_flexibility, prevent_cycles = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0
        start_entities, query_relation,  end_entities, all_answers, batch_weights = data 
        self.no_examples = start_entities.shape[0]
        self.batch_weights = batch_weights
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.weighted_reward = weighted_reward
        self.sigmoid = sigmoid
        self.size_flexibility = size_flexibility
        self.prevent_cycles = prevent_cycles
        self.adjust_factor = adjust_factor
        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        # CREATE A DONE MASK (ONE ENTRY PER ROLLOUT IN THE BATCH)
        self.done_mask = np.zeros(self.no_examples * self.num_rollouts, dtype=bool)


        # Initialize visited entities tracking - each rollout will have its own list
        # Initially, just the starting entities
        self.visited_entities = np.zeros((self.no_examples * self.num_rollouts, 1), dtype=np.int32)
        self.visited_entities[:, 0] = self.start_entities

        self.weight_history = []
        next_actions, next_weights = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, self.visited_entities, self.prevent_cycles)
        
    
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1] # Relations
        self.state['next_entities'] = next_actions[:, :, 0] # Target Entities
        self.state['current_entities'] = self.current_entities # Current Entities
        self.state['weights'] = next_weights # EDGE WEIGHTS
        self.state['visited_entities'] = self.visited_entities 

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward


    def get_reward_weights_sigmoid(self):
        """
        CALCULATE REWARD BASED ON THE POSITIVE REWARD AND THE AVERAGE WEIGHT (IC).
        USE '2.0' AS A SENTINEL FOR PADDING AND IGNORE IT IN THE MEAN.
        """
        # 1) CONVERT THE LIST OF WEIGHT VECTORS INTO A 2D ARRAY: [TIME, BATCH]
        weights_array = np.array(self.weight_history)  # SHAPE (T, B), WHERE T = # STEPS

        # 2) MAKE A MASK FOR THE PADDING (WHERE WE HAVE 2.0) 
        #    'True' => IT IS PADDING, 'False' => REAL WEIGHT
        mask_2 = (weights_array == 2.0)

        # 3) REPLACE 2.0 BY np.nan SO WE CAN IGNORE IT IN THE AVERAGE
        weights_array[mask_2] = np.nan

        # 4) CALCULATE THE MEAN ALONG AXIS=0 (ROLLOUT DIM), IGNORING NaN
        average_ic = np.nanmean(weights_array, axis=0)  # SHAPE [B,]

        # 5) CALCULATE SIZE OF THE PATHS
        size = np.sum(~mask_2, axis=0)  # shape (B,)

       # 6) GIVE A PENALTY TO THE SIZE OF THE PATH:
        #    IF SIZE >= 3 => 0.5 (PENALIZE)
        #    ELSE         => 1 (KEEP REWARD)
        punish_size = np.where(size >= 3, 0.5, 1)
    
        # 7) BUILD THE REWARD BASED ON SUCCESS
        success_mask = (self.current_entities == self.end_entities)

        # 8) CALCULATE REWARD
        if self.sigmoid==True and self.weighted_reward==True:
            positive_part = punish_size * self.positive_reward * average_ic

        elif self.weighted_reward==True and self.sigmoid==False:
            positive_part = self.positive_reward * average_ic
            
        else:
            positive_part = self.positive_reward

        # 9) BUILD THE FINAL REWARD
        condlist   = [success_mask, ~success_mask]
        choicelist = [positive_part, self.negative_reward]
        final_reward = np.select(condlist, choicelist)
        return final_reward


    def get_reward_weights(self):
        """
        Calculate reward based on the positive reward and the weights of the edges.
         - If the current entities match the end entities, reward is calculated as:
        average of edge weights multiplied by the positive reward.
        - Otherwise, the negative reward is applied.
        """
        reward = (self.current_entities == self.end_entities)

        #calculate the average of the edge weights
        average_ic = np.mean(self.weight_history, axis=0) 

        if self.weighted_reward:
            #simple multiplication of the positive reward with the average of the edge weights
            positive_reward = self.positive_reward * average_ic

        else:
            positive_reward = self.positive_reward

        
        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  

        # print if positive reward is given 
        if np.any(reward == self.positive_reward):
            print("Positive reward was given")

        return reward


    def __call__(self, action):
        if self.size_flexibility:
            self.current_hop += 1
            bsz = self.no_examples * self.num_rollouts

            # GET CHOSEN ENTITIES
            chosen_ents = self.state['next_entities'][np.arange(bsz), action]
            self.current_entities = chosen_ents

            # GET THE CHOSEN WEIGHTS
            chosen_weights = self.state['weights'][np.arange(bsz), action]

            # ANY ROLLOUT THAT REACHES THE END_ENTITY => done_mask = TRUE
            newly_done = (chosen_ents == self.end_entities)
            self.done_mask = np.logical_or(self.done_mask, newly_done)

            # FOR ANY ROLLOUT THAT IS DONE, SET THE WEIGHT TO '2.0' TO SIGNAL PADDING
            chosen_weights[self.done_mask] = 2.0

            # APPEND THIS VECTOR OF WEIGHTS (SHAPE [B,]) TO THE HISTORY
            self.weight_history.append(chosen_weights)

            # UPDATE VISITED ENTITIES
            new_visited = np.zeros((bsz, self.visited_entities.shape[1] + 1), dtype=np.int32)
            for i in range(bsz):
                new_visited[i, :self.visited_entities.shape[1]] = self.visited_entities[i]
                new_visited[i, -1] = chosen_ents[i]
            self.visited_entities = new_visited

            
            # GET NEXT ACTIONS/WEIGHTS (WE STILL NEED THIS FOR THE ROLLOUTS NOT DONE)
            next_actions, next_weights = self.grapher.return_next_actions(
                self.current_entities,
                self.start_entities,
                self.query_relation,
                self.end_entities,
                self.all_answers,
                (self.current_hop == self.path_len - 1),
                self.num_rollouts, 
                self.visited_entities, # Pass the visited entities list
                self.prevent_cycles

            )
     
            
            # UPDATE STATE
            self.state['next_relations']  = next_actions[:, :, 1]
            self.state['next_entities']   = next_actions[:, :, 0]
            self.state['current_entities'] = self.current_entities
            self.state['weights'] = next_weights

            return self.state
            

        else:
            self.current_hop += 1
            bsz = self.no_examples * self.num_rollouts

            self.current_entities = self.state['next_entities'][np.arange(bsz), action]

            # Append weights and update next actions as before
            self.weight_history.append(self.state['weights'][np.arange(bsz), action])

            # Update visited entities
            new_visited = np.zeros((bsz, self.visited_entities.shape[1] + 1), dtype=np.int32)
            for i in range(bsz):
                new_visited[i, :self.visited_entities.shape[1]] = self.visited_entities[i]
                new_visited[i, self.visited_entities.shape[1]] = self.current_entities[i]
                    
            self.visited_entities = new_visited

            next_actions, next_weights = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                            self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                            self.num_rollouts, self.visited_entities,  self.prevent_cycles )

            self.state['next_relations'] = next_actions[:, :, 1]
            self.state['next_entities'] = next_actions[:, :, 0]
            self.state['current_entities'] = self.current_entities
            self.state['weights'] = next_weights # EDGE WEIGHTS
            self.state['visited_entities'] = self.visited_entities  # Add visited entities to state

            return self.state


class env(object):
    def __init__(self, params, mode='train'):
        self.weighted_reward = params['weighted_reward']
        self.adjust_factor = params['IC_importance']
        self.sigmoid = params['sigmoid']
        self.size_flexibility = params['size_flexibility']
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.prevent_cycles = params['prevent_cycles']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        input_dir = params['data_input_dir']
        if mode == 'train':
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 edges_weight=params['edges_weight']
                                                 )
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 mode =mode,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 edges_weight=params['edges_weight'])

            self.total_no_examples = self.batcher.store.shape[0]
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'],
                                             edges_weight=params['edges_weight']
                                             )

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher, self.weighted_reward, self.adjust_factor, self.sigmoid, self.size_flexibility, self.prevent_cycles
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
