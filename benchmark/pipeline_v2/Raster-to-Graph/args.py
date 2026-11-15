import argparse

import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    # I/O
    parser.add_argument('--input_dir', default='./input', type=str,
                        help='folder with the floorplan images to process')
    parser.add_argument('--output_dir', default='./output/inference', type=str,
                        help='folder where the visualized results are written')
    parser.add_argument('--resume', default='./output/trained_r2gmodel/checkpoint0299.pth',
                        help='path to the trained checkpoint')
    parser.add_argument('--allow_unsafe_load', action='store_true',
                        help='load the checkpoint with full pickle (only for trusted files)')
    parser.add_argument('--device', default='cuda', help='CUDA device to use, e.g. cuda or cuda:1')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='data loading workers (keep 0 on Windows)')
    parser.add_argument('--seed', default=42, type=int)

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_maps', default=4, type=int, help='number of feature levels')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes for object detection')
    parser.add_argument('--num_classes_edges', default=14, type=int)

    # Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=500, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=20, type=int)
    parser.add_argument('--enc_n_points', default=20, type=int)

    return parser
