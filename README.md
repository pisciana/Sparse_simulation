<h1>Passo a passo para a simulacao da aprendizagem com esparsialidade no supercomputador NPAD </h1>

Arquivos utilizados para a simulação:

*ana_clusterscript01

*process_cluster_npad.py

*script_create_slurm.py 


*script_create_input.py

*simulacao.py

*modelo.py

<b>passo-a-passo simulacao</b>
1) Gerar os arquivos slurm através do script_create_slurm.py 

2) Copiar todos os *.slurm para o servidor

3) No diretório dos arquivos slurm. Digitar o comando: dos2unix *.slurm

4) Gerar os arquivos de input, através do script_create_input.py 

5) Copiar todos os input_files gerados para o mesmo diretorio do slurm no servidor

6) Executar cada slurm com o comando: sbach nome_slurm.slurm

7) Para verificar se o ṕrocesso está executando, utilizar o comando: squeue -u nome_do_usuario



<h1>Passo a passo para exibir resultados em gráfico</h1>
Arquivos utilizados para gerar os gráficos:

