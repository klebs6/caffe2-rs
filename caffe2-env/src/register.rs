crate::ix!();

caffe_register_device_type!{CPU, CPUOperatorRegistry}

caffe_register_device_type!{CUDA, CUDAOperatorRegistry}


caffe_register_device_type!{HIP, HIPOperatorRegistry}

register_creator!{
    /*
    ThreadPoolRegistry,
    CPU,
    GetAsyncNetThreadPool<TaskThreadPool, PROTO_CPU>
    */
}

register_creator!{
    /*
    ThreadPoolRegistry,
    CUDA,
    GetAsyncNetThreadPool<TaskThreadPool, PROTO_CUDA>
    */
}

register_creator!{
    /*
    ThreadPoolRegistry,
    HIP,
    GetAsyncNetThreadPool<TaskThreadPool, PROTO_HIP>
    */
}


