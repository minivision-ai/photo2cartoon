from models import UgatitSadalinHourglass
import argparse
import shutil
from utils import *


def parse_args():
    """parsing and configuration"""
    desc = "photo2cartoon"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=True, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='photo2cartoon', help='dataset name')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=50, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')
    parser.add_argument('--faceid_weight', type=int, default=1, help='Weight for Face ID')

    parser.add_argument('--ch', type=int, default=32, help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    # parser.add_argument('--device', type=str, default='cuda:0', help='Set gpu mode: [cpu, cuda]')
    parser.add_argument('--gpu_ids', type=int, default=[0], nargs='+', help='Set [0, 1, 2, 3] for multi-gpu training')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--rho_clipper', type=float, default=1.0)
    parser.add_argument('--w_clipper', type=float, default=1.0)
    parser.add_argument('--pretrained_weights', type=str, default='', help='pretrained weight path')

    args = parser.parse_args()
    args.result_dir = './experiment/{}-size{}-ch{}-{}-lr{}-adv{}-cyc{}-id{}-identity{}-cam{}'.format(
        os.path.basename(__file__)[:-3],
        args.img_size,
        args.ch,
        args.light,
        args.lr,
        args.adv_weight,
        args.cycle_weight,
        args.faceid_weight,
        args.identity_weight,
        args.cam_weight)

    return check_args(args)


def check_args(args):
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))
    shutil.copy(__file__, args.result_dir)
    return args


def main():
    args = parse_args()
    if args is None:
        exit()

    gan = UgatitSadalinHourglass(args)
    gan.build_model()

    if args.phase == 'train':
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        gan.test()
        print(" [*] Test finished!")


if __name__ == '__main__':
    main()
