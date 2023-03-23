crate::ix!();

register_caffe2_init_function!{
    caffe2_check_intrinsics_features,
    caffe2_check_intrinsics_features,
    "Check intrinsics compatibility between the CPU feature and the binary."
}

register_caffe2_init_function!{
    caffe2_set_denormals,
    caffe2_set_denormals,
    "Set denormal settings."
}

#[cfg(caffe2_use_mkl)]
register_caffe2_init_function!{
    Caffe2SetMKLThreads,
    &Caffe2SetMKLThreads,
    "Set MKL threads."
}

#[cfg(openmp)]
register_caffe2_init_function!{
    Caffe2SetOpenMPThreads,
    &Caffe2SetOpenMPThreads,
    "Set OpenMP threads."
}
