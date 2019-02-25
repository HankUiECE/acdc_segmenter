import numpy as np
import os
from utils import get_best_model_list
from subprocess import Popen
from subprocess import check_output
import subprocess
import shlex
import io

model_path = 'acdc_logdir/unet2D_bn_modified_wxent_bn_hanchao/' 
best_lst = get_best_model_list(model_path, 'model_best_dice.ckpt')
max_dice = 0.0
for number in best_lst:
    command = 'python evaluate_patients.py acdc_logdir/unet2D_bn_modified_wxent_bn_hanchao/ -i ' + number
    proc = Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in io.TextIOWrapper(proc.stderr, encoding="utf-8"):
        idx = line.find('Mean dice')
        if idx != -1:
            line = line.strip('\n')
            dice = float(line[idx + 11:])
            print('current iter: %s, current_dice: %.6f'%(number, dice))
    if dice > max_dice:
        max_dice = dice
        best_iter = number
print('the best iter is: %s, the max_dice is:%.6f'%(best_iter, max_dice))
'''
    info = r.readlines()
    print('total lines ' + str(len(info)))
    for line in info:
        idx = line.find('Mean dice')
        if idx != -1:
           line = line.strip('\r\n')
           dice = line[idx + 11:]
           print('the dice is:' + str(dice))

#command = 'ping -c 5 www.baidu.com'
command = 'python test.py'
command = shlex.split(command)
proc = Popen(command, stdout=subprocess.PIPE)
lst = []
for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
    lst.append(line)
print(lst)
'''
