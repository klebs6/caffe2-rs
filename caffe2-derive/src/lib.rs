extern crate proc_macro;

use proc_macro::TokenStream;
//use quote::quote;
//use syn;

#[proc_macro_attribute]
pub fn test_context(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn __global__(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_OPERATOR_CONTEXT_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_OPERATOR_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_DEFORMABLE_CONV_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn __ubsan_ignore_undefined__(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn compile_warning(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn launch_bounds(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn no_copy(attr: TokenStream, item: TokenStream) -> TokenStream {
    /*
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    impl_hello_macro(&ast)
    */
    item
}

#[proc_macro_attribute]
pub fn noreturn(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn C10_HOST_CONSTEXPR(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_CONV_POOL_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_DISPATCH_HELPER(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_SIMPLE_CTOR_DTOR(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_RECURRENT_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    macro_rules! USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS {
        ($Context:ident) => {
            todo!();
            /*
            USE_OPERATOR_FUNCTIONS(Context);                        
            using ConvTransposeUnpoolBase<Context>::kernel_;        
            using ConvTransposeUnpoolBase<Context>::kernel_h;       
            using ConvTransposeUnpoolBase<Context>::kernel_w;       
            using ConvTransposeUnpoolBase<Context>::stride_;        
            using ConvTransposeUnpoolBase<Context>::stride_h;       
            using ConvTransposeUnpoolBase<Context>::stride_w;       
            using ConvTransposeUnpoolBase<Context>::pads_;          
            using ConvTransposeUnpoolBase<Context>::pad_t;          
            using ConvTransposeUnpoolBase<Context>::pad_l;          
            using ConvTransposeUnpoolBase<Context>::pad_b;          
            using ConvTransposeUnpoolBase<Context>::pad_r;          
            using ConvTransposeUnpoolBase<Context>::adj_;           
            using ConvTransposeUnpoolBase<Context>::group_;         
            using ConvTransposeUnpoolBase<Context>::order_;         
            using ConvTransposeUnpoolBase<Context>::shared_buffer_; 
            using ConvTransposeUnpoolBase<Context>::ws_
            */
        }
    }
    item
}

/**
  | Auxiliary output gradients are currently
  | implemented only for Lengths version
  |
  */
#[macro_export] macro_rules! register_gradient_with_main_input {
    ($gradient_name:expr, $($arg:expr),*) => {
        /*
        
          static_assert(                                                     
              equal(                                                         
                  #gradient_name,                                            
                  __VA_ARGS__::basename,                                     
                  __VA_ARGS__::OpDef::name,                                  
                  "WithMainInputGradient"),                                  
              #gradient_name);                                               
          REGISTER_CPU_OPERATOR_STR(                                         
              string(#gradient_name), __VA_ARGS__::WithMainInputBackwardOp); 
          OPERATOR_SCHEMA(gradient_name)                                     
              .NumInputs(__VA_ARGS__::WithMainInputBackwardOp::kNumInputs)   
              .NumOutputs(1, INT_MAX)
        */
    };
    () => {

    }
}

#[macro_export] macro_rules! 
    register_gradient_with_main_input_and_forward_output {
    () => {
        /*
                (               
            gradient_name, ...)                                                     
          static_assert(                                                            
              equal(                                                                
                  #gradient_name,                                                   
                  __VA_ARGS__::basename,                                            
                  __VA_ARGS__::OpDef::name,                                         
                  "WithMainInputAndForwardOutputGradient"),                         
              #gradient_name);                                                      
          REGISTER_CPU_OPERATOR_STR(                                                
              string(#gradient_name),                                               
              __VA_ARGS__::WithMainInputAndForwardOutputBackwardOp);                
          OPERATOR_SCHEMA(gradient_name)                                            
              .NumInputs(                                                           
                  __VA_ARGS__::WithMainInputAndForwardOutputBackwardOp::kNumInputs) 
              .NumOutputs(1, INT_MAX)
        */
    }
}


#[macro_export] macro_rules! 
    register_segment_def_main_input_and_forward_output_gradient {
    () => {
        /*
                (         
            segment_name, gradient_name, ...)                                        
          static_assert(                                                             
              equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name), 
              #segment_name);                                                        
          OPERATOR_SCHEMA(segment_name)                                              
              .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                         
              .NumOutputs(1)                                                         
              .SetDoc(FormatDoc<__VA_ARGS__>())                                      
              .Output(0, "OUTPUT", "Aggregated tensor")                              
              .FillUsing(__VA_ARGS__::PopulateSchema);                               
          REGISTER_GRADIENT_WITH_MAIN_INPUT_AND_FORWARD_OUTPUT(                      
              gradient_name, __VA_ARGS__);                                           
          REGISTER_GRADIENT_STR(string(#segment_name), __VA_ARGS__::GetGradient)
        */
    }
}

/**
  | This implements and registers a length
  | op with a gradient which requires the
  | main input as well as the output of the
  | forward output.
  |
  */
#[macro_export] macro_rules! 
    register_lengths_ops_main_input_and_forward_output_gradient {
    ($($arg:expr),*) => {
        /*
                (         
            segment_name, gradient_name, ...)                                        
          static_assert(                                                             
              equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name), 
              #segment_name);                                                        
          REGISTER_CPU_OPERATOR_STR(string(#segment_name), __VA_ARGS__::ForwardOp);  
          REGISTER_SEGMENT_DEF_MAIN_INPUT_AND_FORWARD_OUTPUT_GRADIENT(               
              segment_name, gradient_name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! disallow_input_filler { 
    ($a:ident) => {

    }
}

#[macro_export] macro_rules! value_key_length_input_fillers { 
    () => {

    }
}

#[macro_export] macro_rules! fill_using { 
    () => {

    }
}

#[macro_export] macro_rules! 
  weighted_value_key_length_input_fillers { 
    () => {

    }
}

#[macro_export] macro_rules! allow_one_to_one_inplace { 
    ($t:ty) => {

    }
}

#[macro_export] macro_rules! disallow_input_fillers { 
    ($a:ident) => {

    }
}

#[macro_export] macro_rules! scalar_type { 
    ($a:ident, $b:ty) => {

    }
}

#[macro_export] macro_rules! inherit_onnx_schema { 
    ($a:ident) => {

    };
    ($a:expr) => {

    };
    ($a:ident, $d:expr) => {

    }
}

#[macro_export] macro_rules! private_operator { 
    ($a:ident) => {

    }
}

#[macro_export] macro_rules! register_opt_pass_from_func {
    ($passname:ident, $funcname:ident) => {
        /*
        
          class passname : public OptimizationPass {            
           public:                                              
            using OptimizationPass::OptimizationPass;           
            void run() override {                               
              funcname(nn_);                                    
            }                                                   
          };                                                    
          REGISTER_OPT_PASS(passname);
        */
    }
}


#[macro_export] macro_rules! register_ws_opt_pass_from_func {
    ($passname:ident, $funcname:ident) => {
        /*
        
          class passname : public WorkspaceOptimizationPass {           
           public:                                                      
            using WorkspaceOptimizationPass::WorkspaceOptimizationPass; 
            void run() override {                                       
              funcname(nn_, ws_);                                       
            }                                                           
          };                                                            
          REGISTER_WS_OPT_PASS(passname);
        */
    }
}

#[macro_export] macro_rules! identical_type_and_shape_of_input { 
    ($a:ident, $b:expr) => {

    }
}

#[macro_export] macro_rules! identical_type_and_shape_of_input_dim { 
    ($a:ident, ($b:expr, $c:expr)) => {

    }
}

#[macro_export] macro_rules! device_inference_function { 
    () => {

    };
    ($name:ident, $fn:expr) => {

    }
}

#[macro_export] macro_rules! instantiate_test_case_p { 
    () => {

    }
}

#[macro_export] macro_rules! typed_test_case { 
    ($a:ident, $b:ident) => {

    }
}

#[macro_export] macro_rules! export_c10_op_to_caffe2_cpu { 
    ($s:expr, $y:ident $(<$($t:ty),*>)?) => {

    }
}

#[macro_export] macro_rules! c10_register_creator { 
    () => {
        //TODO: elim branch
    };
    ($x:tt, $y:tt, $z:tt) => {

    }
}

#[macro_export] macro_rules! export_caffe2_op_to_c10_cpu { 
    ($x:ty, $s:expr, $y:ty) => {

    }
}

#[macro_export] macro_rules! declare_export_caffe2_op_to_c10 { 
    ($n:tt) => {

    }
}

#[macro_export] macro_rules! export_caffe2_op_to_c10_schema_only { 
    ($s:ident, $y:expr) => {

    }
}

#[macro_export] macro_rules! register_typed_class { 
    ($x:expr, $type:expr, $($arg:expr),*) => {

    }
}

#[macro_export] macro_rules! register_init_function { 
    () => {

    }
}

#[macro_export] macro_rules! register_copy_bytes_function { 
    ($($args:expr),*) => {

    }
}

#[macro_export] macro_rules! register_context {

    () => {/*TODO remove*/};

    ($type:expr, $($arg:expr),*) => {
        register_typed_class!{ContextRegistry, $type, $($arg),*}
    }
}

#[macro_export] macro_rules! caffe_known_type{
    ($ty:ty) => {

    }
}

#[macro_export] macro_rules! register_creator{
    () => {

    }
}

#[macro_export] macro_rules! define_shared_registry {
    () => {

    }
}
#[macro_export] macro_rules! define_string {
    ($name:ident, $default_value:expr, $help_str:expr) => {

    }
}

#[macro_export] macro_rules! declare_string {
    ($name:ident) => {

    }
}

#[macro_export] macro_rules! declare_double {
    ($name:ident) => {

    }
}

#[macro_export] macro_rules! declare_int {
    ($name:ident) => {

    }
}

#[macro_export] macro_rules! declare_int32 {
    ($name:ident) => {

    }
}

#[macro_export] macro_rules! declare_bool {
    ($name:ident) => {

    }
}

#[macro_export] macro_rules! static_assert {
    ($condition:expr, $error_msg:expr) => {

    }
}

#[macro_export] macro_rules! register_transform {
    ($a:tt, $b:tt) => {

    }
}

#[macro_export] macro_rules! define_int {
    ($name:ident, $b:expr, $description:expr) => {

    }
}

#[macro_export] macro_rules! define_double {
    ($name:ident, $b:expr, $description:expr) => {

    }
}

#[macro_export] macro_rules! define_float {
    ($name:ident, $b:expr, $description:expr) => {

    }
}

#[macro_export] macro_rules! define_int64 {
    ($name:ident, $b:expr, $description:expr) => {

    }
}

#[macro_export] macro_rules! define_int32 {
    ($name:ident, $b:expr, $description:expr) => {

    }
}

#[macro_export] macro_rules! inputs_can_cross_devices {
    ($opname:ident) => {

    }
}

#[macro_export] macro_rules! num_inputs {
    ($t:ty, ($min:expr, $max:expr)) => {

    };
    ($t:ty, {$min:expr, $max:expr}) => {

    };
    ($t:ty, $n:expr) => {

    }
}

#[macro_export] macro_rules! num_outputs {
    ($t:ty, ($min:expr, $max:expr)) => {

    };
    ($t:ty, {$min:expr, $max:expr}) => {

    };
    ($t:ty, $n:expr) => {

    }
}

#[macro_export] macro_rules! allow_inplace {
    ($name:ident, $pair_set:expr) => {

    }
}

#[macro_export] macro_rules! identical_type_and_shape_of_multiple_inputs {
    ($name:ident, $pair_set:expr) => {

    }
}

#[macro_export] macro_rules! enforce_inplace {
    ($name:ident, ) => {
        //TODO: eliminate branch

    };

    ($name:ident, $pair_set:expr) => {

    }
}

#[macro_export] macro_rules! enforce_one_to_one_inplace {
    ($x:ty) => {

    }
}

#[macro_export] macro_rules! define_bool {
    ($name:ident, $b:expr, $description:expr) => {

    }
}

#[macro_export] macro_rules! arg_is_test {
    ($argname:ident, $descrip:expr) => {

    }
}

#[macro_export] macro_rules! args {
    ($op:ident, $($idx:expr => ($name:expr, $desc:expr)),*) => {

    }
}

#[macro_export] macro_rules! outputs {
    ($op:ident, $($idx:expr => ($name:expr, $desc:expr)),*) => {

    }
}

#[macro_export] macro_rules! inputs {
    ($op:ident, $($idx:expr => ($name:expr, $desc:expr)),*) => {

    }
}

#[macro_export] macro_rules! args_are_test {
    ($op:ident, $($idx:expr => ($desc:expr)),*) => {

    }
}

#[macro_export] macro_rules! declare_registry {
    ($($args:ty),*) => {

    }
}

#[macro_export] macro_rules! define_registry {
    ($($args:expr),*) => {

    }
}

#[macro_export] macro_rules! declare_shared_registry {
    () => {

    };
    ($registry_name:ident, $object_type:ty, $($arg:ty),*) => {

    }
}

#[macro_export] macro_rules! declare_typed_registry {
    () => {

    };
    ($registry_name:ident, 
        $src_type:ty, 
        $object_type:ty, 
        $ptr_type:ty, 
        $($arg:ty),*) => 
    {

    }
}

#[macro_export] macro_rules! define_typed_registry {
    () => {

    };
    ($registry_name:ident, 
        $src_type:ty, 
        $object_type:ty, 
        $ptr_type:ty, 
        $($arg:ty),*) => 
    {

    }
}

#[macro_export] macro_rules! define_typed_registry_without_warning {
    ($RegistryName:ident, $SrcType:ident, $ObjectType:ident, $PtrType:ident, $($arg:ident),*) => {
        /*
        
          C10_EXPORT ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*    
          RegistryName() {                                                            
            static ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*      
                registry =                                                            
                    new ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>( 
                        false);                                                       
            return registry;                                                          
          }
        */
    }
}

#[macro_export] macro_rules! cost_inference_function {
    ($name:ident,) => {
        //TODO: elim branch
    };
    ($name:ident,$fn:expr) => {

    }
}

#[macro_export] macro_rules! same_number_of_output {
    ($name:ident) => {

    }
}

#[macro_export] macro_rules! output_calculator {
    ($name:ident,) => {

    }
}

#[macro_export] macro_rules! num_inputs_outputs {
    ($name:ident,$fn:expr) => {

    }
}

#[macro_export] macro_rules! tensor_inference_function {
    ($name:ident,) => {
        //TODO: elim branch
    };
    ($name:ident, $fn:expr) => {

    }
}

#[macro_export] macro_rules! identical_type_and_shape {
    ($name:ident) => {

    }
}

#[macro_export] macro_rules! register_event_create_function {
    ($t:ident, $f:ident) => {
        /*
        
          namespace {                                                    
          static EventCreateFunctionRegisterer<t> g_event_create_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_event_record_function {
    ($t:ident, $f:ident) => {
        /*
        
          namespace {                                                    
          static EventRecordFunctionRegisterer<t> g_event_record_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_event_wait_function {
    ($w:ident, $d:ident, $f:ident) => {
        /*
        
          namespace {                                                         
          static EventWaitFunctionRegisterer<w, d> g_event_wait_##w##_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_event_query_function {
    ($t:ident, $f:ident) => {
        /*
        
          namespace {                                                  
          static EventQueryFunctionRegisterer<t> g_event_query_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_event_error_message_function {
    ($t:ident, $f:ident) => {
        /*
        
          namespace {                                                           
          static EventErrorMessageFunctionRegisterer<t> g_event_err_msg_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_event_set_finished_function {
    ($t:ident, $f:ident) => {
        /*
        
          namespace {                                                               
          static EventSetFinishedFunctionRegisterer<t> g_event_set_finished_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_event_set_callback_function {
    ($t:ident, $f:ident) => {
        /*
        
          namespace {                                                               
          static EventSetCallbackFunctionRegisterer<t> g_event_set_callback_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_event_finish_function {
    ($t:ident, $f:ident) => {
        /*
        
          namespace {                                                    
          static EventFinishFunctionRegisterer<t> g_event_finish_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_event_reset_function {
    ($t:ident, $f:ident) => {
        /*
        
          namespace {                                                  
          static EventResetFunctionRegisterer<t> g_event_reset_##d(f); 
          }
        */
    }
}

#[macro_export] macro_rules! register_blob_serializer {
    () => {
        //TODO remove branch
    };
    ($id:ident, $($args:ident),*) => {

        register_typed_class!(BlobSerializerRegistry, $id, $($args),*);

        /// Creates an operator with the given operator definition.
        #[inline] pub fn create_serializer(id: TypeIdentifier) -> Box<BlobSerializerBase> {
            
            todo!();
            /*
                return BlobSerializerRegistry()->Create(id);
            */
        }
    }
}

#[macro_export] macro_rules! register_blob_deserializer {
    () => {
        //TODO remove branch
    };
    ($name:ident, $($arg:ident),*) => {

        register_class!(BlobDeserializerRegistry, name, __VA_ARGS__);

        /// Creates an operator with the given operator definition.
        #[inline] pub fn create_deserializer(ty: &String) -> Box<BlobDeserializerBase> {
            
            todo!();
            /*
                return BlobDeserializerRegistry()->Create(type);
            */
        }
    }
}

#[macro_export] macro_rules! register_caffe2_init_function {
    ($name:ident, 
    $function:ident, 
    $description:expr) => 
    {
        /*
        namespace {                                                              
            ::caffe2::InitRegisterer                                                 
                g_caffe2_initregisterer_##name(function, false, description, #name); 
        } // namespace
        */
    }
}

#[macro_export] macro_rules! register_caffe2_early_init_function {
    ($name:ident, 
    $function:ident, 
    $description:ident) => 
    {
        /*
        namespace {                                                             
            ::caffe2::InitRegisterer                                                
                g_caffe2_initregisterer_##name(function, true, description, #name); 
        } // namespace
        */
    }
}

#[macro_export] macro_rules! register_net_creator {
    ($key:ty, $($arg:ty),*) => {
        /*
        C10_REGISTER_CREATOR(NetRegistry, key, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_net {
    ($name:ty, $($arg:ty),*) => {
        /*
        C10_REGISTER_CLASS(NetRegistry, name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_cpu_operator_creator {
    ($key:ty, $($arg:ty),*) => {
        todo!();
        /*
        
          C10_REGISTER_CREATOR(CPUOperatorRegistry, key, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_cpu_operator {
    ($name:ty, $($arg:ty),*) => {
        /*
        
          C10_IMPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();  
          static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_CPU##name() { 
            CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                
          }                                                                
          C10_REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_cpu_operator_str {
    ($str_name:expr, $($arg:ty),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS(CPUOperatorRegistry, str_name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_cpu_operator_with_engine {
    ($name:ty, 
     $engine:ty, 
     $($arg:ty),*) => {
        /*
        
          C10_REGISTER_CLASS(CPUOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)
        */
    }
}

/**
  | Use these macros to register gradient
  | operators.
  | 
  | They can be automatically excluded
  | from builds that don't need them (e.g.,
  | mobile).
  |
  */
#[cfg(caffe2_no_gradient_ops)]
#[macro_export] macro_rules! register_cpu_gradient_operator {
    ($($arg:ty),*) => {
        /*
                /* No gradients. */
        */
    }
}

#[cfg(not(caffe2_no_gradient_ops))]
#[macro_export] macro_rules! register_cpu_gradient_operator {
    ($($arg:ty),*) => {
        /*
        
          C10_MACRO_EXPAND(REGISTER_CPU_OPERATOR(__VA_ARGS__))
        */
    }
}

#[cfg(caffe2_no_gradient_ops)]
#[macro_export] macro_rules! register_cpu_gradient_operator_with_engine {
    ($($arg:ty),*) => {
        /*
                /* No gradients. */
        */
    }
}

#[cfg(not(caffe2_no_gradient_ops))]
#[macro_export] macro_rules! register_cpu_gradient_operator_with_engine {
    ($($arg:ty),*) => {
        /*
        
          C10_MACRO_EXPAND(REGISTER_CPU_OPERATOR_WITH_ENGINE(__VA_ARGS__))
        */
    }
}


#[macro_export] macro_rules! register_cuda_operator_creator {
    ($key:ty, $($arg:ty),*) => {
        /*
        
          C10_REGISTER_CREATOR(CUDAOperatorRegistry, key, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_cuda_operator {
    ($name:ty, $($arg:ty),*) => {
        /*
        
          C10_IMPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();   
          static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_CUDA##name() { 
            CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                 
          }                                                                 
          C10_REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_cuda_operator_str {
    ($str_name:ty, $($arg:ty),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS(CUDAOperatorRegistry, str_name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_cuda_operator_with_engine {
    ($name:ty, $engine:ty, $($arg:ty),*) => {
        /*
        
          C10_REGISTER_CLASS(CUDAOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)
        */
    }
}

/// Macros for cudnn since we use it often
#[macro_export] macro_rules! register_cudnn_operator {
    ($name:ty, $($arg:ty),*) => {
        /*
        
          REGISTER_CUDA_OPERATOR_WITH_ENGINE(name, CUDNN, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_gradient {

    ($name:expr, $($arg:expr),*) => {

        /*
        #[cfg(caffe2_no_gradient_ops)]
        {
            /*
                    /* No gradients. */
            */
        }

        #[cfg(not(caffe2_no_gradient_ops))]
        {
            /*
            
              C10_REGISTER_CLASS(GradientRegistry, name, __VA_ARGS__)
            */
        }
        */
    };
}

#[cfg(caffe2_no_gradient_ops)]
#[macro_export] macro_rules! register_gradient_str {
    ($str_name:ident, $($arg:ident),*) => {
        /*
                /* No gradients. */
        */
    }
}

#[cfg(not(caffe2_no_gradient_ops))]
#[macro_export] macro_rules! register_gradient_str {
    ($str_name:ident, $($arg:ident),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS(GradientRegistry, str_name, __VA_ARGS__)
        */
    }
}

/**
  | NO_GRADIENT means that the operator
  | does not need any gradient computation.
  |
  */
#[macro_export] macro_rules! no_gradient {
    ($name:ident) => {
        /*
                REGISTER_GRADIENT(name, NoGradient)
        */
    }
}

/**
  | SHOULD_NOT_DO_GRADIENT means that the operator
  | is not designed to have gradient operators. If
  | you attempt to call the gradient, a log fatal
  | will occur.
  */
#[macro_export] macro_rules! should_not_do_gradient {
    ($name:ident) => {
        /*
        
          REGISTER_GRADIENT(name, ThrowInTheTowelIfGradientIsCalled)
        */
    }
}


#[macro_export] macro_rules! gradient_not_implemented_yet {
    ($name:ident) => {
        /*
        
          REGISTER_GRADIENT(name, GradientNotImplementedYet)
        */
    }
}

#[macro_export] macro_rules! register_converter {
    ($name:ident, $cls:ident) => {
        /*
        
          C10_REGISTER_CLASS(ConverterRegistry, name, cls)
        */
    }
}


#[macro_export] macro_rules! trivial_converter {
    ($opName:ident) => {
        /*
        
          class opName##Converter : public Converter {                                
            std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator( 
                const OperatorDef& op) override {                                     
              return std::make_unique<nom::repr::opName>();                     
            }                                                                         
            virtual ~opName##Converter() {}                                           
          };
        */
    }
}

#[macro_export] macro_rules! register_ws_opt_pass {
    ($clsname:ident) => {
        todo!();
        /*
        
          C10_REGISTER_CLASS(WorkspaceOptimizationPassRegistry, clsname, clsname)
        */
    }
}

#[macro_export] macro_rules! register_opt_pass {
    ($clsname:ident) => {
        todo!();
        /*
        
          C10_REGISTER_CLASS(OptimizationPassRegistry, clsname, clsname)
        */
    }
}

#[macro_export] macro_rules! caffe_register_device_type {
    ($type:expr, $registry_function:expr) => {
        /*
        
          namespace {                                               
          static DeviceTypeRegisterer C10_ANONYMOUS_VARIABLE(       
              DeviceType)(type, &registry_function);                
          }
        */
    }
}

#[macro_export] macro_rules! register_ideep_operator_creator {
    ($key:expr, $($arg:expr),*) => {
        /*
        C10_REGISTER_CREATOR(IDEEPOperatorRegistry, key, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_ideep_operator {
    ($name:expr, $($arg:expr),*) => {
        /*
        C10_REGISTER_CLASS(IDEEPOperatorRegistry, name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_ideep_operator_with_engine {
    ($name:expr, $engine:expr, $($arg:expr),*) => {
        /*
        C10_REGISTER_CLASS(IDEEPOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_ideep_operator_str {
    ($str_name:expr, $($arg:expr),*) => {
        /*
        C10_REGISTER_TYPED_CLASS(IDEEPOperatorRegistry, str_name, __VA_ARGS__)
        */
    }
}

#[macro_export] macro_rules! register_ideep_compare_operator {
    ($Op:expr) => {
        /*
        REGISTER_IDEEP_OPERATOR(
            Op,
            IDEEPFallbackOp<
                BinaryElementwiseOp<
                    TensorTypes< bool, int32_t, int64_t, float, double>,  
                    CPUContext,
                    Op##Functor<CPUContext>,
                    FixedType<bool>
                >
            >)
        */
    }
}

#[macro_export] macro_rules! use_ideep_operator_functions {
    () => {
        todo!();
        /*
        USE_OPERATOR_BASE_FUNCTIONS;                                                 
            /* using override */ using IDEEPOperator::Input;                             
            /* using override */ using IDEEPOperator::Output;                            
            /* using override */ using IDEEPOperator::order_;                            
            /* using override */ using IDEEPOperator::context_;
            */
    }
}

#[macro_export] macro_rules! use_simple_ideep_ctor_dtor {
    ($name:ident) => {
        todo!();
        /*
        name(const OperatorDef& operator_def, Workspace* ws)
            : IDEEPOperator(operator_def, ws) {}
        */
    }
}

#[macro_export] macro_rules! is_little_endian {
    () => {
        /*
          [] {                                                        \
            const int32_t kValue = 1;                                 \
            return reinterpret_cast<const uint8_t*>(&kValue)[0] == 1; \
          }()
        */
    }
}

/**
  | OP_SINGLE_ARG provides a shorter initialization
  | choice for initialization of member
  | variables for the class constructors.
  | 
  | This is a workaround for CUDA9.2 and
  | GCC7
  |
  */
#[macro_export] macro_rules! op_single_arg {
    ($type:ident, 
    $name:ident, 
    $variable:ident, 
    $default:ident) => 
    {
        todo!();
        /*
        
          variable(this->template GetSingleArgument<type>(name, (default)))
        */
    }
}

/**
 | INPUT_TAGS and OUTPUT_TAGS are optional features
 | to name the indices of the operator's inputs and
 | outputs, in order to avoid confusion. 
 |
 | For example, for a fully convolution layer that
 | has input, weight and bias, you can define its
 | input tags as:
 |
 |     INPUT_TAGS(INPUT, WEIGHT, BIAS);
 | And in the code, instead of doing
 |     auto& weight = Input(1);
 | you can now do
 |     auto& weight = Input(WEIGHT);
 | to make it more clear.
 */
#[macro_export] macro_rules! input_tags {
    ($first_input:tt { $($args:tt),* } ) => {
        /*
        
          enum _InputTags { first_input = 0, __VA_ARGS__ }
        */
    }
}

#[macro_export] macro_rules! output_tags {
    ($first_input:tt { $($args:tt),* } ) => {
        /*
        
          enum _OutputTags { first_input = 0, __VA_ARGS__ }
        */
    }
}

#[macro_export] macro_rules! reduction_op_shape_inference {
    ($is_front_reducer:ident) => {
        todo!();
        /*
        
          CAFFE_ENFORCE_LE(1, in.size());                                           
          CAFFE_ENFORCE_GE(2, in.size());                                           
          ArgumentHelper helper(def);                                               
          int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1); 
          int start_index = is_front_reducer ? num_reduce_dims : 0;                 
          int end_index = is_front_reducer ? in[0].dims_size()                      
                                           : in[0].dims_size() - num_reduce_dims;   
          vector<int> output_shape;                                                 
          for (int i = start_index; i < end_index; ++i) {                           
            output_shape.push_back(in[0].dims(i));                                  
          }                                                                         
          return vector<TensorShape>{                                               
              CreateTensorShape(output_shape, in[0].data_type())};
        */
    }
}

