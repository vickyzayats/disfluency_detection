import sys, os
import socket

config_file = 'configs/attention_multi_sim_conv.py'
forward_lm = '../lm/best_models/lm_forward/77.1338_85.4927_19_4000.h5'
backward_lm = '../lm/best_models/lm_backward/49.6490_52.7537_12_6000.h5'
num_exps = 2

filename = os.path.basename(config_file).replace('.py','.txt')

with open(filename , 'w') as outf:
    for i in range(20/num_exps):
        outf.write('bash run_disfl_jobs.sh %d %s %d %s %s' % \
                   (num_exps, config_file, i, forward_lm, backward_lm) + '\n')
