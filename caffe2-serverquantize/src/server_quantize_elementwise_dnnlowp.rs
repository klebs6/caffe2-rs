crate::ix!();

use crate::{
    CPUContext,
    OperatorDef,
    Operator,
    Workspace,
    OperatorStorage,
    DNNLowPOp,
    RequantizationParams,
};

pub struct UnaryElementwiseWithArgsDNNLowPOp<T,Functor> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    storage: OperatorStorage,
    context: CPUContext,

    functor:           Functor,
    arguments_parsed:  bool, // default = false
    phantom: PhantomData<T>,
}

impl<T,Functor> UnaryElementwiseWithArgsDNNLowPOp<T,Functor> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws), functor_()
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (!arguments_parsed_) {
          dnnlowp::ParseDNNLowPOperatorArguments(this);
          dnnlowp::SetStaticQuantizationParams(
              this, 0, functor_.GetOutputQuantizationParams());
          arguments_parsed_ = true;
        }

        auto& input = this->template Input<int8::Int8TensorCPU>(0).t;
        auto& output = Outputs()[0]->template GetMutable<int8::Int8TensorCPU>()->t;
        output.ResizeLike(input);
        functor_(
            input.size(),
            input.template data<T>(),
            output.template mutable_data<T>());

        dnnlowp::PropagateOutputTensorQuantizationParams(
            this, 0, functor_.GetOutputQuantizationParams());
        return true;
        */
    }
}

///--------------
pub struct BinaryElementwiseDNNLowPOp<T,FP32_OP> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    base: DNNLowPOp<T, FP32_OP>,

    enable_broadcast:       bool,
    axis:                   i32,
    axis_str:               String,
    order:                  String,
    requantization_params:  RequantizationParams,
}

impl<T, FP32_OP> BinaryElementwiseDNNLowPOp<T,FP32_OP> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : DNNLowPOp<T, FP32_OP>(operator_def, ws),
            OP_SINGLE_ARG(bool, "broadcast", enable_broadcast_, 0),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
            OP_SINGLE_ARG(string, "order", order_, "NCHW") 

        // Figure out the correct axis to use.
        if (enable_broadcast_) {
          if (axis_ != -1) {
            // Get axis from an explicit axis argument.
            CAFFE_ENFORCE_EQ(
                axis_str_.size(),
                0,
                "Args axis and axis_str cannot be used simultaneously.");
          } else if (axis_str_.size()) {
            // Get the axis index semantically.
            CAFFE_ENFORCE_EQ(
                axis_str_.size(), 1, "Unsupported axis string", axis_str_);
            size_t semantic_axis_ = order_.find(axis_str_);
            CAFFE_ENFORCE_NE(
                semantic_axis_,
                string::npos,
                "Unrecognizable axis string ",
                axis_str_,
                " from order string ",
                order_);
            axis_ = semantic_axis_;
          }
        } else {
          CAFFE_ENFORCE(
              axis_ == -1 && axis_str_.size() == 0,
              "Do not specify axis or axis_str if broadcast is not enabled.");
        }
        */
    }
}

/**
  | For arithmetic operators, Eigen provides
  | a good way to vectorize even when broadcasting.
  |
  */
#[macro_export] macro_rules! declare_eigen_functor {
    ($name:ident, $eigen_op:ident, $input_type:ident, $output_type:ident) => {
        todo!();
        /*
        
          struct Eigen##name##Functor {                                              
            template <int b_is_scalar, typename T, typename R>                       
            inline void Run(size_t n, const T* a, const T* b, R* out, CPUContext*) { 
              if (b_is_scalar) {                                                     
                EigenVectorArrayMap<R>(out, n) =                                     
                    eigen_op((ConstEigenVectorArrayMap<T>(a, n)), (b[0]));           
              } else {                                                               
                EigenVectorArrayMap<R>(out, n) = eigen_op(                           
                    (ConstEigenVectorArrayMap<T>(a, n)),                             
                    (ConstEigenVectorArrayMap<T>(b, n)));                            
              }                                                                      
            }                                                                        
            template <typename T, typename R>                                        
            void RunWithBroadcast(                                                   
                const T* a,                                                          
                const T* b,                                                          
                R* out,                                                              
                size_t pre,                                                          
                size_t n,                                                            
                CPUContext*) {                                                       
              EigenArrayMap<R>(out, n, pre) = eigen_op(                              
                  (ConstEigenArrayMap<T>(a, n, pre).colwise()),                      
                  (ConstEigenVectorArrayMap<T>(b, n)));                              
            }                                                                        
            template <typename T, typename R>                                        
            void RunWithBroadcast2(                                                  
                const T* a,                                                          
                const T* b,                                                          
                R* out,                                                              
                size_t pre,                                                          
                size_t n,                                                            
                size_t post,                                                         
                CPUContext*) {                                                       
              for (int i = 0; i < pre; ++i) {                                        
                EigenArrayMap<R>(out + i * n * post, post, n) = eigen_op(            
                    (ConstEigenArrayMap<T>(a + i * n * post, post, n).rowwise()),    
                    (Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>>(b, n)));   
              }                                                                      
            }                                                                        
          };
        */
    }
}
