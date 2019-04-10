export CUDA_VISIBLE_DEVICES=0,1

datadir=./data
vectordir=./word2vec
modeldir=./saved_models
predictdir=./predictions

# change accordingly when choosing different models
model_suffix="CNN_BiLSTM"

#clean the previous runs, careful!!!!
if [ -d "$modeldir" ]; then
  # Control will enter here if $DIRECTORY exists.
  rsync -a $modeldir ${modeldir}_bak
  #rm -rf $modeldir
  #recreate the directories
  #mkdir -p $modeldir
  else
  mkdir -p $modeldir
  
fi

if [ -d "$predictdir" ]; then
  # Control will enter here if $DIRECTORY exists.
  rsync -a $predictdir ${predictdir}_bak
  #rm -rf $predictdir
  #recreate the directories
  #mkdir -p $predictdir
else
  mkdir -p $predictdir
fi

if [ -d log ]; then
  # Control will enter here if $DIRECTORY exists.
  # make copy of results to the back up file
  rsync -a log log_backup 
  #rm -rf log
fi





python main.py \
      --model=${model_suffix} \
      --batch_size=64 \
      --vocab_size=40000 \
      --embed_dim=300 \
      --in_channels=1 \
      --kernel_sizes=345\
      --kernel_nums=200 \
      --hidden_dim=200 \
      --num_layers=2 \
      --bidirectional=True\
      --max_seq_len=60\
      --learning_rate=0.1\
      --num_epochs=100 \
      --dropout=0\
      --num_class=2 \
      --train_file=${datadir}/cwmt_train_comb.txt \
      --dev_file=${datadir}/cwmt_dev_comb.txt \
      --test_file=${datadir}/htqe_test_comb.txt \
      --pretrained_embeddings=${vectordir}/wiki.en_zh.vec \
      --prediction_file=${predictdir}/htqe_test_comb_${model_suffix}.txt \
      --saved_model=${modeldir}/cwmt.en_zh_${model_suffix}.pt \
      --test=False \
      --weight_decay=0.0001\
      --grad_clip=0.5\
      --seed_num=123 \
      --run_log="umtqe_${model_suffix}"
