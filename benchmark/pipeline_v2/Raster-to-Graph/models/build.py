import torch
from models.position_encoding import PositionEmbeddingSine
from models.deformable_detr import DeformableDETR
from models.deformable_transformer import DeformableTransformer
from models.backbone import Joiner, Backbone
from models.postprocess import PostProcess


def build_backbone_with_position_embedding(args):
    position_embedding = PositionEmbeddingSine(args.hidden_dim // 2, normalize=True)
    backbone = Backbone(args.backbone)
    return Joiner(backbone, position_embedding)


def build_deforamble_transformer(args):
    return DeformableTransformer(
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        num_feature_levels=args.num_feature_maps,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points)


def build_model(args, device):
    backbone_with_position_embedding = build_backbone_with_position_embedding(args)
    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone_with_position_embedding,
        transformer,
        num_classes=args.num_classes,
        num_classes_edges=args.num_classes_edges,
        num_queries=args.num_queries,
        num_feature_maps=args.num_feature_maps
    )
    model.to(device)
    return model


def build_postprocessor():
    return PostProcess()
