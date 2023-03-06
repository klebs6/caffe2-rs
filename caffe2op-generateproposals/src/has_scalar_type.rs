/*!
  | Bounding box utils for generate_proposals_op
  | 
  | Reference: facebookresearch/Detectron/detectron/utils/boxes.py
  |
  */

crate::ix!();

/**
  | Default value for minimum bounding box width
  |     and height after bounding box
  |     transformation (bbox_transform()) in
  |     log-space
  */
pub const BBOX_XFORM_CLIP_DEFAULT: f32 = 1.79588; //(1000.0 / 16.0).log10();

/**
 | Forward transform that maps proposal boxes to
 | ground-truth boxes using bounding-box regression
 | deltas.
 |
 | boxes: pixel coordinates of the bounding boxes
 |
 |     size (M, 4), format [x1; y1; x2; y2], x2 >= x1, y2 >= y1
 |
 |     deltas: bounding box translations and scales
 |
 |     size (M, 4), format [dx; dy; dw; dh]
 |
 |     dx, dy: scale-invariant translation of the
 |     center of the bounding box
 |
 |     dw, dh: log-space scaling of the width and
 |     height of the bounding box
 |
 | weights: weights [wx, wy, ww, wh] for the deltas
 |
 | bbox_xform_clip: minimum bounding box width and
 |     height in log-space after transofmration
 |
 | correct_transform_coords: Correct bounding box
 |     transform coordates. Set to true to match the
 |     detectron code, set to false for backward
 |     compatibility
 |
 | return: pixel coordinates of the bounding boxes
 |     size (M, 4), format [x1; y1; x2; y2]
 |
 | see "Rich feature hierarchies for accurate object
 |     detection and semantic segmentation" Appendix
 |     C for more details
 |
 | reference: detectron/lib/utils/boxes.py
 | bbox_transform()
 |
 | const std::vector<typename Derived2::Scalar>&
 | weights = std::vector<typename
 | Derived2::Scalar>{1.0, 1.0, 1.0, 1.0},
 */
pub trait HasScalarType {
    type Scalar: Float;
}
