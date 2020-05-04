import argparse
from Trainer import Trainer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported bool string format.')

def main():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='batch size for evaluating')
    parser.add_argument('--train_list', type=str, default='dataset/train.txt', help='list file for training')
    parser.add_argument('--eval_list', type=str, default='dataset/eval.txt', help='list file for validation')
    parser.add_argument('--img_path', type=str, default='dataset/sat', help='path for images of dataset')
    parser.add_argument('--gt_path', type=str, default='dataset/mask', help='path for ground truth of dataset')
    parser.add_argument('--num_of_class', type=int, default=2, choices=[1,2], help='number of classes')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
    parser.add_argument('--cuda', type=str2bool, default=False, help='whether to use GPU')
    parser.add_argument('--loss', type=str, default='BCE', choices=['CE','BCE','LS','F','CE+D'], help='type of loss function')
    parser.add_argument('--model', type=str, default='deeplabv3+', help='model to train')
    parser.add_argument('--init_eval', type=str2bool, default=False, help='whether to start with evaluation')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of training')
    parser.add_argument('--reproduce', type=str2bool, default=False, help='whether to use given seed')
    parser.add_argument('--submodel', type=str2bool, default=False, help='whether to use submodel')

    args = parser.parse_args()
    print(args)
    if args.num_of_class == 1:
        assert args.loss in ['BCE'], "Loss function & #class do not match."
    else: #args.num_of_class == 2
        assert args.loss in ['LS', 'CE', 'CE+D', 'F'], "Loss function & #class do not match."
    my_trainer = Trainer(args)
    my_trainer.run()

if __name__ == "__main__":
   main()