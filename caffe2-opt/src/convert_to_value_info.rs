crate::ix!();

#[inline] pub fn convert_to_value_info(
    names:             &Vec<String>,
    shape_hints:       &HashMap<String,TensorShape>,
    extra_shape_hints: &HashMap<String,OnnxTypeProto>) -> Vec<OnnxValueInfoProto> {
    
    todo!();
    /*
        std::vector<::ONNX_NAMESPACE::ValueInfoProto> r;
      for (const auto& s : names) {
        r.emplace_back();
        auto& value_info = r.back();
        value_info.set_name(s);
        const auto it = shape_hints.find(s);
        if (it == shape_hints.end()) {
          const auto eit = extra_shape_hints.find(s);
          if (eit == extra_shape_hints.end()) {
            LOG(WARNING) << "Cannot get shape of " << s;
          } else {
            value_info.mutable_type()->CopyFrom(eit->second);
          }
        } else {
          auto* tensor_type = value_info.mutable_type()->mutable_tensor_type();
          tensor_type->set_elem_type(
              OnnxCaffe2TypeToOnnxType(it->second.data_type()));
          auto* shape = tensor_type->mutable_shape();
          for (int i = 0; i < it->second.dims().size(); ++i) {
            shape->add_dim()->set_dim_value(it->second.dims(i));
          }
        }
      }
      return r;
    */
}
