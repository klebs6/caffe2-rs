crate::ix!();

#[inline] pub fn bbox_transform_upright<Derived1: HasScalarType, Derived2: HasScalarType<Scalar = f32>>(
    boxes:           &ArrayBase<Derived1>,
    deltas:          &ArrayBase<Derived2>,
    weights:         Option<&Vec<<Derived2 as HasScalarType>::Scalar>>,
    bbox_xform_clip: Option<f32>,
    legacy_plus_one: Option<bool>) -> EArrXXt<<Derived1 as HasScalarType>::Scalar> 
{
    let bbox_xform_clip:  f32 = bbox_xform_clip.unwrap_or(BBOX_XFORM_CLIP_DEFAULT);
    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);

    let weights = weights.unwrap_or(&vec![1.0,1.0,1.0,1.0]);

    todo!();
    /*
        using T = typename Derived1::Scalar;
      using EArrXX = EArrXXt<T>;
      using EArrX = EArrXt<T>;

      if (boxes.rows() == 0) {
        return EArrXX::Zero(T(0), deltas.cols());
      }

      CAFFE_ENFORCE_EQ(boxes.rows(), deltas.rows());
      CAFFE_ENFORCE_EQ(boxes.cols(), 4);
      CAFFE_ENFORCE_EQ(deltas.cols(), 4);

      EArrX widths = boxes.col(2) - boxes.col(0) + T(int(legacy_plus_one));
      EArrX heights = boxes.col(3) - boxes.col(1) + T(int(legacy_plus_one));
      auto ctr_x = boxes.col(0) + T(0.5) * widths;
      auto ctr_y = boxes.col(1) + T(0.5) * heights;

      auto dx = deltas.col(0).template cast<T>() / weights[0];
      auto dy = deltas.col(1).template cast<T>() / weights[1];
      auto dw =
          (deltas.col(2).template cast<T>() / weights[2]).cwiseMin(bbox_xform_clip);
      auto dh =
          (deltas.col(3).template cast<T>() / weights[3]).cwiseMin(bbox_xform_clip);

      EArrX pred_ctr_x = dx * widths + ctr_x;
      EArrX pred_ctr_y = dy * heights + ctr_y;
      EArrX pred_w = dw.exp() * widths;
      EArrX pred_h = dh.exp() * heights;

      EArrXX pred_boxes = EArrXX::Zero(deltas.rows(), deltas.cols());
      // x1
      pred_boxes.col(0) = pred_ctr_x - T(0.5) * pred_w;
      // y1
      pred_boxes.col(1) = pred_ctr_y - T(0.5) * pred_h;
      // x2
      pred_boxes.col(2) = pred_ctr_x + T(0.5) * pred_w - T(int(legacy_plus_one));
      // y2
      pred_boxes.col(3) = pred_ctr_y + T(0.5) * pred_h - T(int(legacy_plus_one));

      return pred_boxes;
    */
}
