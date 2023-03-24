crate::ix!();

#[inline] pub fn set_device_option(
    n: NNGraph_NodeRef, 
    d: &mut DeviceOption)  {
    
    todo!();
    /*
        getOrAddCaffe2Annotation(n);
      auto op = nn::get<NeuralNetOperator>(n);
      auto c2Annot = dyn_cast<caffe2::Caffe2Annotation>(op->getMutableAnnotation());
      CAFFE_ENFORCE(c2Annot, "getOrAddCaffe2Annotation failed!");
      c2Annot->setDeviceOption(d);
    */
}

