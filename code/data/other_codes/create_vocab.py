import json
import csv
import argparse
import os

dataset = 'primekg'
root_dir = '../../'
vocab_dir = root_dir + 'datasets/' + dataset + '/vocab/'
dir = root_dir + 'datasets/' + dataset  + '/'
os.makedirs(vocab_dir)

entity_vocab = {}
relation_vocab = {}

entity_vocab['PAD'] = len(entity_vocab)
entity_vocab['UNK'] = len(entity_vocab)
relation_vocab['PAD'] = len(relation_vocab)
relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
relation_vocab['UNK'] = len(relation_vocab)

entity_counter = len(entity_vocab)
relation_counter = len(relation_vocab)

for f in ['train.txt', 'dev.txt', 'test.txt', 'graph.txt']:
    with open(dir+f) as raw_file:
        csv_file = csv.reader(raw_file, delimiter='\t')
        for line in csv_file:
            e1,r,e2 = line
            if e1 not in entity_vocab:
                entity_vocab[e1] = entity_counter
                entity_counter += 1
            if e2 not in entity_vocab:
                entity_vocab[e2] = entity_counter
                entity_counter += 1
            if r not in relation_vocab:
                relation_vocab[r] = relation_counter
                relation_counter += 1

with open(vocab_dir + 'entity_vocab.json', 'w') as fout:
    json.dump(entity_vocab, fout)

with open(vocab_dir + 'relation_vocab.json', 'w') as fout:
    json.dump(relation_vocab, fout)


