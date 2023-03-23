crate::ix!();

register_cpu_operator!{ImageInput, ImageInputOp<CPUContext>}

#[cfg(caffe2_use_mkldnn)]
register_ideep_operator!{ImageInput, IDEEPFallbackOp<ImageInputOp<CPUContext>>}

register_cuda_operator!{ImageInput, ImageInputOp<CUDAContext>}
