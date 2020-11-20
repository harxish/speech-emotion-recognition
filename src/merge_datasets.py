import os
from glob import glob

for file in glob('dataset*/Actor_*/*.wav'):

	namee = file.split('/')[-1]
	namee = namee.split('.')
	namee[0] += 'a'; 
	namee = '.'.join(namee)
	file = '/'.join(file.split('/')[:-1]) + '/' + namee
	
	break