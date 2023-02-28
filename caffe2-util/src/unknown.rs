/*!
where are these types?
*/
crate::ix!();

#[derive(Default)]       pub struct CPUGeneratorImpl       { }
pub struct AVCodecContext                                  { }
pub struct AVFrame                                         { }
pub struct AVIOContext                                     { }
pub struct AVMediaType                                     { }
pub struct AVPacket                                        { }
pub struct AVPixelFormat                                   { }
pub struct AveragePoolFp320p                               { } 
pub struct Bookkeeper                                      { }
pub struct CPUOp                                           { }
pub struct CUDAStream                                      { }
pub struct CompressedSparseColumn                          { }
pub struct Context                                         { } 
pub struct CuBlasStatus                                    { }
pub struct CuRandGenerator                                 { }
pub struct CuRandStatus                                    { }
pub struct CublasHandle                                    { }
pub struct CudaDeviceProp                                  { }
pub struct CudaEvent                                       { }
pub struct CudaGuard                                       { }
pub struct CudaStream                                      { }
pub enum CudnnDataType { 
    CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF,
}
pub struct CudnnHandle                                     { }
pub struct CudnnStatus                                     { }
pub struct CudnnTensorDescriptor                           { }
pub struct CudnnTensorFormat                               { }
pub struct DNNLowPOp<A,B>                                  { phantomA: PhantomData<A>, phantomB: PhantomData<B>, }
pub struct Database                                        { }
pub struct DatabaseMode                                    { }
pub struct Dim3                                            { x: i32, y: i32, z: i32, }
pub struct FixedDivisor<T>                                 { phantom: PhantomData<T>, }
pub struct FunctionSchema                                  { }
pub struct Fuser                                           { } 
pub struct IAlgo                                           { }
pub struct IAttr                                           { }
pub struct IDEEPConvolutionForwardParams                   { }
pub struct IDEEPScale                                      { }
pub struct IDEEPTensor                                     { }
pub struct IDEEPTensorDescriptor                           { }
pub struct IDEEPTensorDims                                 { }
pub struct IDType                                          { }
pub struct IFormat                                         { }
pub struct ILowpKind                                       { }
pub struct IProp                                           { }
pub struct IScale                                          { }
pub struct ITensor                                         { }
pub struct ITensorDescriptor                               { }
pub struct ITensorDims                                     { }
pub struct IValue                                          { }
pub struct IntrusivePtr<T>                                 { phantom: PhantomData<T>, }
pub struct LabelType                                       { }
pub struct List<T>                                         { phantom: PhantomData<T>, }
pub struct MaxPoolFp320p                                   { }
pub struct MemoryType                                      { }
pub struct NNGraph_EdgeRef                                 { } 
pub struct NNGraph_NodeRef                                 { } 
pub struct NNGraph_SubgraphType                            { }
pub struct OnnxBackend                                     { } 
pub struct OnnxBackendGraphMap                             { } 
pub struct OnnxBackendID                                   { }
pub struct OnnxExporter                                    { }
pub struct OnnxGraph                                       { } 
pub struct OnnxModelProto                                  { }
pub struct OnnxSharedPtrBackendGraphInfo                   { } 
pub struct OnnxStatus                                      { } 
pub struct OnnxTensorDescriptorV1                          { } 
pub struct OnnxTypeProto                                   { }
pub struct OnnxValueInfoProto                              { }
pub struct OnnxifiLibrary                                  { }
pub struct OperatorHandle                                  { }
pub struct OperatorName                                    { }
pub struct PackBMatrix<A,B=A>                              { phantomA: PhantomData<A>, phantomB: PhantomData<B>, }
pub struct PackWeightMatrixForGConv<T>                     { phantomT: PhantomData<T>, }
pub struct PackWeightMatrixForGConvB<A,B,const N: usize>   { phantomA: PhantomData<A>, phantomB: PhantomData<B>, }
pub struct PackedDepthWiseConvMatrix                       { }
pub struct PatternMatchType                                { }
pub struct PerGpuCudnnStates                               { }
pub struct Range<T>                                        { phantom: PhantomData<T>, }
pub struct RequantizationParams                            { } 
pub struct ScalingParamType                                { }
pub struct SerializationFormat                             { }
pub struct SigAction                                       { }
pub struct SkipOutputCopy                                  { }
pub struct SmallVec                                        { } 
pub struct Stack                                           { }
pub struct SwrContext                                      { }
pub struct Test<T>                                         { phantom: PhantomData<T>, }
pub struct TestMatchGraph_NodeRef                          { }
pub struct TestWithParam<Args>                             { p: PhantomData<Args>, } 
pub struct Types<T>                                        { phantom: PhantomData<T>, }
pub struct ZipArchive                                      { }

pub struct EigenArrayMap<T> {
    phantom: PhantomData<T>,
}

pub struct ConstEigenArrayMap<T> {
    phantom: PhantomData<T>,
}
pub struct CudnnActivationDescriptor {}
pub struct CudnnConvolutionDescriptor {}

pub struct ERArrXXf { }
pub struct EArrXf { }
pub struct EArrXi { }
pub struct EArrXXt<T> { p: PhantomData<T>, }

pub struct CudnnFilterDescriptor         {}
pub struct CudnnConvolutionFwdAlgo       {}
pub struct CudnnConvolutionBwdFilterAlgo {}
pub struct CudnnConvolutionBwdDataAlgo   {}
pub struct GPUContext   {}
pub struct HIPContext   {}
pub struct SimpleQueue<X>   { p: PhantomData<X> }
pub struct CudnnDropoutDescriptor   {}
pub struct CudnnRNNDescriptor   {}
pub struct CudnnLRNDescriptor   {}

pub type Vector2f = [f32; 2];
pub type Vector3f = [f32; 3];
pub type Vector2d = [f64; 2];
pub type Vector3d = [f64; 3];
pub type VectorXd = Vec<f64>;
pub type VectorXf = Vec<f32>;

pub type Array2f = [f32; 2];
pub type Array3f = [f32; 3];
pub type Array2d = [f64; 2];
pub type Array3d = [f64; 3];
pub type ArrayXd = Vec<f64>;
pub type ArrayXf = Vec<f32>;

pub struct QnnpOperator   {}
pub struct CudnnPoolingDescriptor   {}
pub struct CudnnPoolingMode   {}
pub struct FixedType<X>   { p: PhantomData<X> }
pub struct ArrayBase<A>   { 
    p1: PhantomData<A>,
}
pub struct TensorCoreEngine   {}
