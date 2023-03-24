crate::ix!();

#[cfg(not(c10_mobile))]
#[inline] pub fn copy_descriptor(
    from: *const ExternalTensorDescriptor, 
    to:   *mut OnnxTensorDescriptorV1)  
{
    todo!();
    /*
        to->dataType = from->dataType;
      to->buffer = from->buffer;
      to->isOffline = from->isOffline;
      to->quantizationParams = from->quantizationParams;
      to->quantizationAxis = from->quantizationAxis;
      to->scales = from->scales;
      to->biases = from->biases;
      to->dimensions = from->dimensions;
      to->shape = from->shape;
    */
}

