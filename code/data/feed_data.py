from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os


class RelationEntityBatcher():
    def __init__(self, input_dir, batch_size, entity_vocab, relation_vocab, edges_weight, mode = "train"):
        self.input_dir = input_dir
        self.input_file = input_dir+'/{0}.txt'.format(mode)
        self.batch_size = batch_size
        print('Reading vocab...')
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.edges_weight = edges_weight
        self.mode = mode
        self.create_triple_store(self.input_file)
        print("batcher loaded")

    def get_next_batch(self):
        if self.mode == 'train':
            yield self.yield_next_batch_train()
        else:
            yield self.yield_next_batch_test()


    def create_triple_store(self, input_file):
        self.store_all_correct = defaultdict(set)
        self.store = []
        self.weights = []  # NEW STRUCTURE FOR WEIGHTS
        if self.mode == 'train':
            with open(input_file) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                for line in csv_file:
                    ent1, rel, ent2 = line[0], line[1], line[2]
                    e1 = self.entity_vocab[ent1]
                    r = self.relation_vocab[rel]
                    e2 = self.entity_vocab[ent2]

                    # ADD WEIGHTS TO THE EDGES FOR CLUSTERED IC BY EDGE TYPE
                    if rel in self.edges_weight.keys():
                        w_e1 = self.edges_weight[rel].get(ent1, 0.5)
                        w_e2 = self.edges_weight[rel].get(ent2, 0.5)
                        w = (w_e1 + w_e2) / 2
                    else:
                        w = 0.5

                    #store weights 
                    self.store.append([e1,r,e2])
                    self.weights.append(w)  # STORE WEIGHT
                    self.store_all_correct[(e1, r)].add(e2)

            self.store = np.array(self.store)
            self.weights = np.array(self.weights)  # WEIGHTS AS ARRAYS

        else:  
            with open(input_file) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                for line in csv_file:
                    ent1, rel, ent2 = line[0], line[1], line[2]
                    if ent1 in self.entity_vocab and ent2 in self.entity_vocab:
                        e1 = self.entity_vocab[ent1]
                        r = self.relation_vocab[rel]
                        e2 = self.entity_vocab[ent2]

                        # ADD WEIGHTS TO THE EDGES 
                        if rel in self.edges_weight.keys():
                            w_e1 = self.edges_weight[rel].get(ent1, 0.5)
                            w_e2 = self.edges_weight[rel].get(ent2, 0.5)
                            w = (w_e1 + w_e2) / 2
                        else:
                            w = 0.5

                        # store info
                        self.store.append([e1,r,e2])
                        self.weights.append(w)  # STORE WEIGHT

            self.store = np.array(self.store)
            self.weights = np.array(self.weights)  # WEIGHTS AS ARRAY 

            fact_files = ['train.txt', 'test.txt', 'dev.txt', 'graph.txt']

            for f in fact_files:
                with open(self.input_dir+'/'+f) as raw_input_file:
                    csv_file = csv.reader(raw_input_file, delimiter='\t')
                    for line in csv_file:
                        ent1, rel, ent2 = line[0], line[1], line[2]
                        if ent1 in self.entity_vocab and ent2 in self.entity_vocab:
                            e1 = self.entity_vocab[ent1]
                            r = self.relation_vocab[rel]
                            e2 = self.entity_vocab[ent2]
                            self.store_all_correct[(e1, r)].add(e2)


    def yield_next_batch_train(self):
        while True:
            # Select random index for batch
            batch_idx = np.random.randint(0, self.store.shape[0], size=self.batch_size)

            # Extract components from self.store
            batch = self.store[batch_idx, :]
            batch_weights = self.weights[batch_idx]  # SELECT WEIGHTS FROM BATCH

            e1 = batch[:,0] # Source entities
            r = batch[:, 1] # Relations
            e2 = batch[:, 2] # Target entities
            
            # Determine valid target entities for each (e1, r)
            all_e2s = [self.store_all_correct[(e1[i], r[i])] for i in range(e1.shape[0])]
        
            # Ensure all components have consistent sizes
            assert e1.shape[0] == r.shape[0] == e2.shape[0] == len(all_e2s)
            
            yield e1, r, e2, all_e2s, batch_weights
 
    def yield_next_batch_test(self):
        remaining_triples = self.store.shape[0]
        current_idx = 0
        while True:
            # Check if we have finished the epoch
            if remaining_triples == 0:
                return
                
            # Select indices for the next batch
            if remaining_triples - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                batch_idx = np.arange(current_idx, self.store.shape[0])
                remaining_triples = 0

            # Extract components from self.store
            batch = self.store[batch_idx, :]
            batch_weights = self.weights[batch_idx]  # SELECT WEIGHTS FROM BATCH
            
            e1 = batch[:,0] # Source entities
            r = batch[:, 1]  # Relations
            e2 = batch[:, 2] # Target entities

            # Determine valid target entities for each (e1, r)
            all_e2s = [self.store_all_correct[(e1[i], r[i])] for i in range(e1.shape[0])]
            
            # Ensure all components have consistent sizes
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            yield e1, r, e2, all_e2s, batch_weights
