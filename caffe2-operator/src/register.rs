crate::ix!();

#[macro_export] macro_rules! register_hip_operator_creator {
    ($key:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_CREATOR(HIPOperatorRegistry, key, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_hip_operator {
    ($name:ident, $($arg:ident),*) => {
        /*
        
          C10_IMPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();  
          static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_HIP##name() { 
            CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                
          }                                                                
          C10_REGISTER_CLASS(HIPOperatorRegistry, name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_hip_operator_str {
    ($str_name:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS(HIPOperatorRegistry, str_name, __VA_ARGS__)
        */
    }
}


#[macro_export] macro_rules! register_hip_operator_with_engine {
    ($name:ident, $engine:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_CLASS(HIPOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_miopen_operator {
    ($name:ident, $($arg:ident),*) => {
        /*
        
          REGISTER_HIP_OPERATOR_WITH_ENGINE(name, MIOPEN, __VA_ARGS__) 
          REGISTER_HIP_OPERATOR_WITH_ENGINE(                           
              name, CUDNN, __VA_ARGS__) // Make CUDNN an alias of MIOPEN for HIP ops
        */
    }
}


register_creator!{
    /*
    ThreadPoolRegistry,
    CPU,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_CPU>
    */
}

register_creator!{
    /*
    ThreadPoolRegistry,
    CUDA,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_CUDA>
    */
}

register_creator!{
    /*
    ThreadPoolRegistry,
    HIP,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_HIP>
    */
}

register_creator!{
    /*
    TaskGraphRegistry, 
    futures, 
    GetAsyncTaskGraph
    */
}

register_net!{parallel, ParallelNet}
register_net!{simple, SimpleNet}

register_cpu_operator!{NetSimpleRefCountTest, NetSimpleRefCountTestOp}
