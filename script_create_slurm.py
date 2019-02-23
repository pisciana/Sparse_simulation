# -*- coding: utf-8 -*-
# gerar script
import numpy as np

whereisit = "npad";
jobname = "ANA"

callstr = "CAMINHO_PYTHON"
procestr = "CODIGO_A_SER_EXECUTADO"
dirpos = "DIRETORIO_DO_SLURM"


numprocss = 4

for filenumber in np.arange(0,14):
    arquivo = open(dirpos + whereisit + "_%.3d" % filenumber + ".slurm", 'w')

    tosend = "#!/bin/bash\n"  
    tosend += "#SBATCH --job-name=" + jobname + "_%.3d\n" % filenumber
    tosend += "#SBATCH --output=ttt/slurm_" + jobname + "%.3d.out\n" % filenumber
    tosend += "#SBATCH --error=ttt/slurm_" + jobname + "%.3d.err\n" % filenumber
    tosend += "#SBATCH --nodes=1\n"
    tosend += "#SBATCH --ntasks-per-node=32\n"
    tosend += "#SBATCH --time=4-23:59\n\n"
#    tosend += "#SBATCH -x service[1-4]\n\n"
    
    tosend += "export OMP_NUM_THREADS=%d\n\n" % int(32/numprocss)
    tosend += callstr + " " + procestr + " %d %d\n" % (filenumber,numprocss)
    
    arquivo.write(tosend)
    arquivo.close()



