exp='maml-omniglot-5way-1shot-TEST'
dataset='omniglot'
num_cls=5
num_inst=1
batch=1
num_updates=15000
num_inner_updates=1
num_inner_processes=4 # use > 0 for multiprocessing inner loop
m_batch=32
lr='1e-1'
meta_lr='1e-3'
gpu=0

exp="maml-omniglot-5way-1shot"

python maml.py $exp --dataset $dataset \
--num_cls $num_cls --num_inst $num_inst \
--batch $batch --m_batch $m_batch --num_updates $num_updates \
--num_inner_updates $num_inner_updates --lr $lr --meta_lr $meta_lr \
--gpu $gpu --num_inner_processes $num_inner_processes 2>&1 | tee ../logs/$exp
