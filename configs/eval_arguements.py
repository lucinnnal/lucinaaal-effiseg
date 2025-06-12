from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, default="/home/urp_jwl/.vscode-server/data/Effiseg/src/data/cityscapes")
    parser.add_argument('--input-size', type=int, default=512, help='size of the input image to resize => default 512*1024')
    parser.add_argument('--target-size', type=int, default=128, help='size of the target label map to resize => default 128*256')
    parser.add_argument('--weightspath', default='./ckpt/segformerb2_teacher_cityscapes.pth')
    parser.add_argument('--subset', default="val")  
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model', type=str, default='SegformerB2', help='model type to use for evaluation')
    parser.add_argument('--num_classes', type=int, default=20, help='number of classes in the dataset')
    
    return parser.parse_args()