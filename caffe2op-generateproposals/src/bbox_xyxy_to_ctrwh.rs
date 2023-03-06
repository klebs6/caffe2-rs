crate::ix!();

#[inline] pub fn bbox_xyxy_to_ctrwh<Derived: HasScalarType>(
    boxes: &ArrayBase<Derived>, 
    legacy_plus_one: Option<bool>) -> EArrXXt<<Derived as HasScalarType>::Scalar> {

    let legacy_plus_one: bool =
             legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE_EQ(boxes.cols(), 4);

      const auto& x1 = boxes.col(0);
      const auto& y1 = boxes.col(1);
      const auto& x2 = boxes.col(2);
      const auto& y2 = boxes.col(3);

      EArrXXt<typename Derived::Scalar> ret(boxes.rows(), 4);
      ret.col(0) = (x1 + x2) / 2.0; // x_ctr
      ret.col(1) = (y1 + y2) / 2.0; // y_ctr
      ret.col(2) = x2 - x1 + int(legacy_plus_one); // w
      ret.col(3) = y2 - y1 + int(legacy_plus_one); // h
      return ret;
    */
}

#[inline] pub fn bbox_ctrwh_to_xyxy<Derived: HasScalarType>(
    boxes: &ArrayBase<Derived>, 
    legacy_plus_one: Option<bool>) -> EArrXXt<<Derived as HasScalarType>::Scalar> {

    let legacy_plus_one: bool =
             legacy_plus_one.unwrap_or(false);

    todo!();
    /*
        CAFFE_ENFORCE_EQ(boxes.cols(), 4);

      const auto& x_ctr = boxes.col(0);
      const auto& y_ctr = boxes.col(1);
      const auto& w = boxes.col(2);
      const auto& h = boxes.col(3);

      EArrXXt<typename Derived::Scalar> ret(boxes.rows(), 4);
      ret.col(0) = x_ctr - (w - int(legacy_plus_one)) / 2.0; // x1
      ret.col(1) = y_ctr - (h - int(legacy_plus_one)) / 2.0; // y1
      ret.col(2) = x_ctr + (w - int(legacy_plus_one)) / 2.0; // x2
      ret.col(3) = y_ctr + (h - int(legacy_plus_one)) / 2.0; // y2
      return ret;
    */
}
