import argparse
import sys

sys.path.append('..')
import VDSR.vdsr as vdsr
import VDSR.train as train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='-')
    parser.add_argument('--training_epoch', type=int, default=80, help='-')
    parser.add_argument('--batch_size', type=int, default=64, help='-')
    parser.add_argument('--n_channel', type=int, default=1, help='-')
    parser.add_argument('--grad_clip', type=float, default=1e-1, help='-')
    parser.add_argument('--on_grad_clipping', type=bool, default=False, help='-')
    args, unknown = parser.parse_known_args()

    VDSR = vdsr.VDSR(args)
    train.training(VDSR, config=args)