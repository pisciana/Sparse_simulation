<h1>Problemas encontrados nas simulações</h1>

<b>1)Cannot create a tensor proto whose content is larger than 2GB</b>
  Tentativa 1) Essa semana tentei executar novamente as simulações, dessa vez com 4
  nucleos, o objetivo era resolver o problema de restrição do TensorFlow com vetores acima
  de 2GB, pois não estava conseguindo gerar os arquivos com a acurácia.

  A simulação ainda está sendo executada, no entanto, verifiquei que continua com o erro:
  ValueError: Cannot create a tensor proto whose content is larger than 2GB.

Tentativa 2) Então alterei o código referente à criação de dados randômicos então o
método ficou:


<b>v2019-02-28 07:41:16.048466: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
terminate called after throwing an instance of 'std::system_error'
  what():  Resource temporarily unavailable</b>
