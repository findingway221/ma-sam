import ants
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fix_path', type=str, required=True, help='the path of fixing image (i.e., the coarse segmentation of input image)')
parser.add_argument('--move_path', type=str, required=True, help='the path of moving image (i.e., the multiple label maps of atlas)')
parser.add_argument('--save_path', type=str, required=True, help='the path to save warped results')
args = parser.parse_args()

fix_img = ants.image_read(args.fix_path)
move_img = ants.image_read(args.move_path)


outs = ants.registration(fixed=fix_img, moving=move_img, type_of_transform='SyN')
warped = outs['warpedmovout']

ants.image_write(warped, args.save_path)

print(f'done!! fix:{args.fix_path} move:{args.move_path}, save to {args.save_path}')