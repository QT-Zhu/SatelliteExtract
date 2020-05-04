python3 train.py --num_of_class 1 --model gcn --cuda False --train_batch_size 2 \
    --eval_batch_size 2 --loss BCE --init_eval False --epochs 200 \
    --resume ./checkpoints/GCN_BCE_epoch160.pth.tar --reproduce True