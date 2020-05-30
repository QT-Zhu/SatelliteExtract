python3 train.py \
    --cuda False \
    --num_of_class 1 \
    --model gcn \
    --loss BCE+D \
    --submodel mobilenet_v2 \
    --selected_layer 0 1 2 3 \
    --epochs 200 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --init_eval False \
    --reproduce True 
    
    