crate::ix!();

/// Wrap TensorShape into TensorProto
#[inline] pub fn wrap_shape_info_into_tensor_proto(
    name: &String,
    shape_info: &ShapeInfo) -> TensorProto 
{
    todo!();
    /*
        TensorProto t;
      t.set_name(name);
      t.set_data_type(shape_info.shape.data_type());
      for (const auto i : shape_info.shape.dims()) {
        t.add_dims(i);
      }
      for (const auto& dimType : shape_info.getDimType()) {
        t.add_int32_data(static_cast<int32_t>(dimType));
      }
      return t;
    */
}

/// Wrap Quantized TensorShape into QTensorProto
#[inline] pub fn wrap_shape_info_into_qtensor_proto(
    name: &String,
    shape_info: &ShapeInfo) -> QTensorProto 
{
    todo!();
    /*
        QTensorProto t;
      CAFFE_ENFORCE(
          shape_info.is_quantized == true,
          "Only quantized shapeinfo can be extracted into QTensor!");
      t.set_name(name);
      t.set_data_type(shape_info.shape.data_type());
      t.set_axis(shape_info.q_info.axis);
      t.set_is_multiparam(true);
      for (const auto i : shape_info.q_info.scale) {
        t.add_scales(i);
      }
      t.set_scale(1.0);
      for (const auto i : shape_info.q_info.offset) {
        t.add_biases(i);
      }
      t.set_bias(0.0);
      // precision and is_signed is not used in onnxifi workflow, but it is required
      // field
      t.set_precision(0);
      t.set_is_signed(0);
      for (const auto i : shape_info.shape.dims()) {
        t.add_dims(i);
      }
      for (const auto& dimType : shape_info.getDimType()) {
        t.add_data(static_cast<int32_t>(dimType));
      }
      return t;
    */
}
