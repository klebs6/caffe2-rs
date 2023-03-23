crate::ix!();

/**
  | Structure to store per-image information
  | 
  | This can be modified by the DecodeAnd*
  | so needs to be privatized per launch.
  |
  */
pub struct PerImageArg {
    bounding_params: BoundingBox,
}

pub struct BoundingBox {
    valid:  bool,
    ymin:   i32,
    xmin:   i32,
    height: i32,
    width:  i32,
}

/**
  | SINGLE_LABEL: single integer label
  | for multi-class classification
  | 
  | MULTI_LABEL_SPARSE: sparse active
  | label indices for multi-label classification
  | MULTI_LABEL_DENSE: dense label embedding
  | vector for label embedding regression
  | MULTI_LABEL_WEIGHTED_SPARSE: sparse
  | active label indices with per-label
  | weights for multi-label classification
  | 
  | SINGLE_LABEL_WEIGHTED: single integer
  | label for multi-class classification
  | with weighted sampling EMBEDDING_LABEL:
  | an array of floating numbers representing
  | dense embedding.
  | 
  | It is useful for model distillation
  |
  */
pub enum ImageInputOpLabelType {
    SingleLabel,
    MultiLabelSparse,
    MultiLabelDense,
    MultiLabelWeightedSparse,
    SingleLabelWeighted,
    EmbeddingLabel,
}

/**
  | INCEPTION_STYLE: Random crop with
  | size 8% - 100% image area and aspect ratio
  | in [3/4, 4/3]. Reference: GoogleNet
  | paper
  |
  */
pub enum ScaleJitterType {
    NoScaleJitter,
    InceptionStyle,
    // TODO(zyan3): ResNet-style random scale jitter
}
