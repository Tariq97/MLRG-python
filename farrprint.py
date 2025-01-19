# Created by Hugo Cruz Jimenez, August 2011, KAUST
import numpy as np
def farrprint(*args):
#function farrprint(fid,in,par)
#
#	function farrprint(fid,in,'par') is used to
#	write 2D-array as ascii to disk with the option
#	of labeling each row; usually called in a loop
#	over one of the array-dimensions
#	
#	INPUT
#	
#	fid 	- file identifier from earlier call to FOPEN
#	in  	- vector to write to disk
#	'par' 	- optional string for labeling rows 
#
#	mmai, 10/26/99
#	--------------

    if len(args) == 2:
        fid=args[0]
        iin=args[1]
        par= ''
    elif len(args) == 3:
        fid=args[0]
        iin=args[1]
        par=args[2]

    if len(iin) != 0:
    
        k = len(iin)
        
        for i in range(0,k):
        
            if i == 0: 
                fid.write('{0} {1:8.2f}'.format(par,iin[i]))
            elif (i < k-1 and i > 0) :
                fid.write('{0:8.2f}'.format(iin[i]))
            elif i == k-1:
                fid.write('{0:8.2f}\n'.format(iin[i]))
    
    elif len(iin) == 0:
      
        fid.write('{0} \n'.format(par))
