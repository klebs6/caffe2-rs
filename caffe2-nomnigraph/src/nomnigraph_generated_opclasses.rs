crate::ix!();

use crate::{
    NeuralNetOperator
};

pub struct Relu {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Relu);
    base: NeuralNetOperator,
}

impl Default for Relu {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Relu
        */
    }
}

///-------------------------------

pub struct Conv {
    base: NeuralNetOperator,
    //NOMNIGRAPH_DEFINE_NN_RTTI(Conv);

    kernel_shape:  Vec<i32>,
    pads:          Vec<i32>,
    strides:       Vec<i32>,
    group:         i32,
    dilations:     Vec<i32>,
}

impl Conv {
    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>,
        group:           Option<i32>,
        dilations:       Option<Vec<i32>>) -> Self 
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);
        let group     = group.unwrap_or(1);
        let dilations = dilations.unwrap_or(vec![1, 1]);

        todo!();
        /*
      : NeuralNetOperator(NNKind::Conv),
        kernelShape_(kernelShape),
        pads_(pads),
        strides_(strides),
        group_(group),
        dilations_(dilations) 
        */
    }
    
    #[inline] pub fn get_kernel_shape(&self) -> Vec<i32> {
        
        todo!();
        /*
            return kernelShape_;
        */
    }
    
    #[inline] pub fn get_pads(&self) -> Vec<i32> {
        
        todo!();
        /*
            return pads_;
        */
    }
    
    #[inline] pub fn get_strides(&self) -> Vec<i32> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn get_group(&self) -> i32 {
        
        todo!();
        /*
            return group_;
        */
    }
    
    #[inline] pub fn get_dilations(&self) -> Vec<i32> {
        
        todo!();
        /*
            return dilations_;
        */
    }
    
    #[inline] pub fn set_kernel_shape(&mut self, kernel_shape: Vec<i32>)  {
        
        todo!();
        /*
            kernelShape_ = std::move(kernelShape);
        */
    }
    
    #[inline] pub fn set_pads(&mut self, pads: Vec<i32>)  {
        
        todo!();
        /*
            pads_ = std::move(pads);
        */
    }
    
    #[inline] pub fn set_strides(&mut self, strides: Vec<i32>)  {
        
        todo!();
        /*
            strides_ = std::move(strides);
        */
    }
    
    #[inline] pub fn set_group(&mut self, group: i32)  {
        
        todo!();
        /*
            group_ = group;
        */
    }
    
    #[inline] pub fn set_dilations(&mut self, dilations: Vec<i32>)  {
        
        todo!();
        /*
            dilations_ = std::move(dilations);
        */
    }
}


///-----------------------------
pub struct ConvRelu {
    //NOMNIGRAPH_DEFINE_NN_RTTI(ConvRelu);
    base: NeuralNetOperator,

    kernel_shape:  Vec<i32>,
    pads:          Vec<i32>,
    strides:       Vec<i32>,
    group:         i32,
    dilations:     Vec<i32>,
}

impl From<&Conv> for ConvRelu {

    fn from(conv: &Conv) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::ConvRelu),
            kernelShape_(conv.getKernelShape()),
            pads_(conv.getPads()),
            strides_(conv.getStrides()),
            group_(conv.getGroup()),
            dilations_(conv.getDilations())
        */
    }
}

impl ConvRelu {

    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>,
        group:           Option<i32>,
        dilations:       Option<Vec<i32>>) -> Self 
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);
        let group     = group.unwrap_or(1);
        let dilations = dilations.unwrap_or(vec![1, 1]);

        todo!();
        /*
          : NeuralNetOperator(NNKind::ConvRelu),
            kernelShape_(kernelShape),
            pads_(pads),
            strides_(strides),
            group_(group),
            dilations_(dilations) 
            */
    }
    
    #[inline] pub fn get_kernel_shape(&self) -> Vec<i32> {
        
        todo!();
        /*
            return kernelShape_;
        */
    }
    
    #[inline] pub fn get_pads(&self) -> Vec<i32> {
        
        todo!();
        /*
            return pads_;
        */
    }
    
    #[inline] pub fn get_strides(&self) -> Vec<i32> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn get_group(&self) -> i32 {
        
        todo!();
        /*
            return group_;
        */
    }
    
    #[inline] pub fn get_dilations(&self) -> Vec<i32> {
        
        todo!();
        /*
            return dilations_;
        */
    }
    
    #[inline] pub fn set_kernel_shape(&mut self, kernel_shape: Vec<i32>)  {
        
        todo!();
        /*
            kernelShape_ = std::move(kernelShape);
        */
    }
    
    #[inline] pub fn set_pads(&mut self, pads: Vec<i32>)  {
        
        todo!();
        /*
            pads_ = std::move(pads);
        */
    }
    
    #[inline] pub fn set_strides(&mut self, strides: Vec<i32>)  {
        
        todo!();
        /*
            strides_ = std::move(strides);
        */
    }
    
    #[inline] pub fn set_group(&mut self, group: i32)  {
        
        todo!();
        /*
            group_ = group;
        */
    }
    
    #[inline] pub fn set_dilations(&mut self, dilations: Vec<i32>)  {
        
        todo!();
        /*
            dilations_ = std::move(dilations);
        */
    }
}

///-------------------------
pub struct ConvTranspose {
    //NOMNIGRAPH_DEFINE_NN_RTTI(ConvTranspose);
    base: NeuralNetOperator,

    kernel_shape:  Vec<i32>,
    pads:          Vec<i32>,
    strides:       Vec<i32>,
    group:         i32,
    dilations:     Vec<i32>,
}

impl ConvTranspose {
    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>,
        group:           Option<i32>,
        dilations:       Option<Vec<i32>>) -> Self 
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);
        let group     = group.unwrap_or(1);
        let dilations = dilations.unwrap_or(vec![1, 1]);

        todo!();
        /*
          : NeuralNetOperator(NNKind::ConvTranspose),
            kernelShape_(kernelShape),
            pads_(pads),
            strides_(strides),
            group_(group),
            dilations_(dilations) 
        */
    }

    #[inline] pub fn get_kernel_shape(&self) -> Vec<i32> {
        
        todo!();
        /*
            return kernelShape_;
        */
    }
    
    #[inline] pub fn get_pads(&self) -> Vec<i32> {
        
        todo!();
        /*
            return pads_;
        */
    }
    
    #[inline] pub fn get_strides(&self) -> Vec<i32> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn get_group(&self) -> i32 {
        
        todo!();
        /*
            return group_;
        */
    }
    
    #[inline] pub fn get_dilations(&self) -> Vec<i32> {
        
        todo!();
        /*
            return dilations_;
        */
    }
    
    #[inline] pub fn set_kernel_shape(&mut self, kernel_shape: Vec<i32>)  {
        
        todo!();
        /*
            kernelShape_ = std::move(kernelShape);
        */
    }
    
    #[inline] pub fn set_pads(&mut self, pads: Vec<i32>)  {
        
        todo!();
        /*
            pads_ = std::move(pads);
        */
    }
    
    #[inline] pub fn set_strides(&mut self, strides: Vec<i32>)  {
        
        todo!();
        /*
            strides_ = std::move(strides);
        */
    }
    
    #[inline] pub fn set_group(&mut self, group: i32)  {
        
        todo!();
        /*
            group_ = group;
        */
    }
    
    #[inline] pub fn set_dilations(&mut self, dilations: Vec<i32>)  {
        
        todo!();
        /*
            dilations_ = std::move(dilations);
        */
    }
}


///------------------------------------
pub struct AveragePool {
    //NOMNIGRAPH_DEFINE_NN_RTTI(AveragePool);
    base: NeuralNetOperator,

    kernel_shape: Vec<i32>,
    pads:         Vec<i32>,
    strides:      Vec<i32>,
}
impl AveragePool {
    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>) -> Self
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);

        todo!();
        /*
          : NeuralNetOperator(NNKind::AveragePool),
            kernelShape_(kernelShape),
            pads_(pads),
            strides_(strides) 
        */
    }
    
    #[inline] pub fn get_kernel_shape(&self) -> Vec<i32> {
        
        todo!();
        /*
            return kernelShape_;
        */
    }
    
    #[inline] pub fn get_pads(&self) -> Vec<i32> {
        
        todo!();
        /*
            return pads_;
        */
    }
    
    #[inline] pub fn get_strides(&self) -> Vec<i32> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn set_kernel_shape(&mut self, kernel_shape: Vec<i32>)  {
        
        todo!();
        /*
            kernelShape_ = std::move(kernelShape);
        */
    }
    
    #[inline] pub fn set_pads(&mut self, pads: Vec<i32>)  {
        
        todo!();
        /*
            pads_ = std::move(pads);
        */
    }
    
    #[inline] pub fn set_strides(&mut self, strides: Vec<i32>)  {
        
        todo!();
        /*
            strides_ = std::move(strides);
        */
    }
}

///----------------------
pub struct AveragePoolRelu {
    //NOMNIGRAPH_DEFINE_NN_RTTI(AveragePoolRelu);
    base: NeuralNetOperator,

    kernel_shape: Vec<i32>,
    pads:         Vec<i32>,
    strides:      Vec<i32>,
}

impl From<&AveragePool> for AveragePoolRelu {

    fn from(average_pool: &AveragePool) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::AveragePoolRelu),
            kernelShape_(averagePool.getKernelShape()),
            pads_(averagePool.getPads()),
            strides_(averagePool.getStrides())
        */
    }
}

impl AveragePoolRelu {
    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>) -> Self
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);

        todo!();
        /*
          : NeuralNetOperator(NNKind::AveragePoolRelu),
            kernelShape_(kernelShape),
            pads_(pads),
            strides_(strides) 
        */
    }

    #[inline] pub fn get_kernel_shape(&self) -> Vec<i32> {
        
        todo!();
        /*
            return kernelShape_;
        */
    }
    
    #[inline] pub fn get_pads(&self) -> Vec<i32> {
        
        todo!();
        /*
            return pads_;
        */
    }
    
    #[inline] pub fn get_strides(&self) -> Vec<i32> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn set_kernel_shape(&mut self, kernel_shape: Vec<i32>)  {
        
        todo!();
        /*
            kernelShape_ = std::move(kernelShape);
        */
    }
    
    #[inline] pub fn set_pads(&mut self, pads: Vec<i32>)  {
        
        todo!();
        /*
            pads_ = std::move(pads);
        */
    }
    
    #[inline] pub fn set_strides(&mut self, strides: Vec<i32>)  {
        
        todo!();
        /*
            strides_ = std::move(strides);
        */
    }
}


///---------------------------------------
pub struct MaxPool {
    //NOMNIGRAPH_DEFINE_NN_RTTI(MaxPool);
    base: NeuralNetOperator,

    kernel_shape: Vec<i32>,
    pads:         Vec<i32>,
    strides:      Vec<i32>,
}

impl MaxPool {

    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>) -> Self
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);

        todo!();
        /*
          : NeuralNetOperator(NNKind::MaxPool),
            kernelShape_(kernelShape),
            pads_(pads),
            strides_(strides) 
        */
    }
    
    #[inline] pub fn get_kernel_shape(&self) -> Vec<i32> {
        
        todo!();
        /*
            return kernelShape_;
        */
    }
    
    #[inline] pub fn get_pads(&self) -> Vec<i32> {
        
        todo!();
        /*
            return pads_;
        */
    }
    
    #[inline] pub fn get_strides(&self) -> Vec<i32> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn set_kernel_shape(&mut self, kernel_shape: Vec<i32>)  {
        
        todo!();
        /*
            kernelShape_ = std::move(kernelShape);
        */
    }
    
    #[inline] pub fn set_pads(&mut self, pads: Vec<i32>)  {
        
        todo!();
        /*
            pads_ = std::move(pads);
        */
    }
    
    #[inline] pub fn set_strides(&mut self, strides: Vec<i32>)  {
        
        todo!();
        /*
            strides_ = std::move(strides);
        */
    }
}

///---------------------------------
pub struct MaxPoolRelu {
    //NOMNIGRAPH_DEFINE_NN_RTTI(MaxPoolRelu);
    base: NeuralNetOperator,

    kernel_shape: Vec<i32>,
    pads:         Vec<i32>,
    strides:      Vec<i32>,
}

impl From<&MaxPool> for MaxPoolRelu {

    fn from(max_pool: &MaxPool) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::MaxPoolRelu),
            kernelShape_(maxPool.getKernelShape()),
            pads_(maxPool.getPads()),
            strides_(maxPool.getStrides())
        */
    }
}

impl MaxPoolRelu {

    pub fn new(
        kernel_shape:    Vec<i32>, 
        pads:            Option<Vec<i32>>,
        strides:         Option<Vec<i32>>) -> Self
    {
        let pads      = pads.unwrap_or(vec![0, 0]);
        let strides   = strides.unwrap_or(vec![1, 1]);

        todo!();
        /*
          : NeuralNetOperator(NNKind::MaxPoolRelu),
            kernelShape_(kernelShape),
            pads_(pads),
            strides_(strides) 
        */
    }
    
    #[inline] pub fn get_kernel_shape(&self) -> Vec<i32> {
        
        todo!();
        /*
            return kernelShape_;
        */
    }
    
    #[inline] pub fn get_pads(&self) -> Vec<i32> {
        
        todo!();
        /*
            return pads_;
        */
    }
    
    #[inline] pub fn get_strides(&self) -> Vec<i32> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn set_kernel_shape(&mut self, kernel_shape: Vec<i32>)  {
        
        todo!();
        /*
            kernelShape_ = std::move(kernelShape);
        */
    }
    
    #[inline] pub fn set_pads(&mut self, pads: Vec<i32>)  {
        
        todo!();
        /*
            pads_ = std::move(pads);
        */
    }
    
    #[inline] pub fn set_strides(&mut self, strides: Vec<i32>)  {
        
        todo!();
        /*
            strides_ = std::move(strides);
        */
    }
}

///---------------------
pub struct Sum {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Sum);
    base: NeuralNetOperator,
}

impl Default for Sum {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Sum
        */
    }
}


///--------------------------
pub struct SumRelu {
    //NOMNIGRAPH_DEFINE_NN_RTTI(SumRelu);
    base: NeuralNetOperator,
}

impl Default for SumRelu {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::SumRelu
        */
    }
}

impl From<&Sum> for SumRelu {
    
    fn from(sum: &Sum) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::SumRelu)
        */
    }
}


///--------------------
pub struct Send_ {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Send_);
    base: NeuralNetOperator,
    destination: String,
}

impl Send_ {

    pub fn new(destination: String) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Send_), destination_(destination)
        */
    }
    
    #[inline] pub fn get_destination(&self) -> String {
        
        todo!();
        /*
            return destination_;
        */
    }
    
    #[inline] pub fn set_destination(&mut self, destination: String)  {
        
        todo!();
        /*
            destination_ = std::move(destination);
        */
    }
}


///--------------------------
pub struct Receive {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Receive);
    base: NeuralNetOperator,
    source: String,
}

impl Receive {

    pub fn new(source: String) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Receive), source_(source)
        */
    }
    
    #[inline] pub fn get_source(&self) -> String {
        
        todo!();
        /*
            return source_;
        */
    }
    
    #[inline] pub fn set_source(&mut self, source: String)  {
        
        todo!();
        /*
            source_ = std::move(source);
        */
    }
}

///---------------------------------
pub struct BatchNormalization {
    //NOMNIGRAPH_DEFINE_NN_RTTI(BatchNormalization);
    base: NeuralNetOperator,

    epsilon:  f32,
    momentum: f32,
    spatial:  bool,
    is_test:  bool,
}

impl BatchNormalization {
    
    pub fn new(
        epsilon:  Option<f32>,
        momentum: Option<f32>,
        spatial:  Option<bool>,
        is_test:  Option<bool>) -> Self {

        let epsilon:  f32 = epsilon.unwrap_or(1e-5);
        let momentum: f32 = momentum.unwrap_or(0.9);
        let spatial: bool = spatial.unwrap_or(true);
        let is_test: bool = is_test.unwrap_or(false);

        todo!();
        /*
            : NeuralNetOperator(NNKind::BatchNormalization),
            epsilon_(epsilon),
            momentum_(momentum),
            spatial_(spatial),
            isTest_(isTest)
        */
    }
    
    #[inline] pub fn get_epsilon(&self) -> f32 {
        
        todo!();
        /*
            return epsilon_;
        */
    }
    
    #[inline] pub fn get_momentum(&self) -> f32 {
        
        todo!();
        /*
            return momentum_;
        */
    }
    
    #[inline] pub fn get_spatial(&self) -> bool {
        
        todo!();
        /*
            return spatial_;
        */
    }
    
    #[inline] pub fn get_is_test(&self) -> bool {
        
        todo!();
        /*
            return isTest_;
        */
    }
    
    #[inline] pub fn set_epsilon(&mut self, epsilon: f32)  {
        
        todo!();
        /*
            epsilon_ = epsilon;
        */
    }
    
    #[inline] pub fn set_momentum(&mut self, momentum: f32)  {
        
        todo!();
        /*
            momentum_ = momentum;
        */
    }
    
    #[inline] pub fn set_spatial(&mut self, spatial: bool)  {
        
        todo!();
        /*
            spatial_ = spatial;
        */
    }
    
    #[inline] pub fn set_is_test(&mut self, is_test: bool)  {
        
        todo!();
        /*
            isTest_ = isTest;
        */
    }
}

///---------------------
pub struct Clip {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Clip);
    base: NeuralNetOperator,

    min: f32,
    max: f32,
}

impl Clip {

    pub fn new(min: f32, max: f32) -> Self {
    
        todo!();
        /*
            : NeuralNetOperator(NNKind::Clip), min_(min), max_(max)
        */
    }
    
    #[inline] pub fn get_min(&self) -> f32 {
        
        todo!();
        /*
            return min_;
        */
    }
    
    #[inline] pub fn get_max(&self) -> f32 {
        
        todo!();
        /*
            return max_;
        */
    }
    
    #[inline] pub fn set_min(&mut self, min: f32)  {
        
        todo!();
        /*
            min_ = min;
        */
    }
    
    #[inline] pub fn set_max(&mut self, max: f32)  {
        
        todo!();
        /*
            max_ = max;
        */
    }
}

///--------------------------

pub struct FC {
    //NOMNIGRAPH_DEFINE_NN_RTTI(FC);
    base: NeuralNetOperator,

    axis:  i32,
    axisW: i32,
}

impl FC {

    pub fn new(
        axis: Option<i32>, 
        axisW: Option<i32>) -> Self 
    {
        let axis: i32 = axis.unwrap_or(1);
        let axisW: i32 = axisW.unwrap_or(1);

        todo!();
        /*
            : NeuralNetOperator(NNKind::FC), axis_(axis), axisW_(axisW)
        */
    }
    
    #[inline] pub fn get_axis(&self) -> i32 {
        
        todo!();
        /*
            return axis_;
        */
    }
    
    #[inline] pub fn get_axisW(&self) -> i32 {
        
        todo!();
        /*
            return axisW_;
        */
    }
    
    #[inline] pub fn set_axis(&mut self, axis: i32)  {
        
        todo!();
        /*
            axis_ = axis;
        */
    }
    
    #[inline] pub fn set_axisW(&mut self, axisW: i32)  {
        
        todo!();
        /*
            axisW_ = axisW;
        */
    }
}

///-------------------
pub struct GivenTensorFill {
    //NOMNIGRAPH_DEFINE_NN_RTTI(GivenTensorFill);
    base: NeuralNetOperator,
}

impl Default for GivenTensorFill {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::GivenTensorFill
        */
    }
}

///----------------------
pub struct Concat {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Concat);
    base: NeuralNetOperator,

    axis:     i32,
    add_axis: bool,
}

impl Concat {
    
    pub fn new(
        axis: Option<i32>, 
        add_axis: Option<bool>) -> Self {

        let axis: i32 = axis.unwrap_or(-1);
        let add_axis: bool = add_axis.unwrap_or(false);

        todo!();
        /*
            : NeuralNetOperator(NNKind::Concat), axis_(axis), addAxis_(addAxis)
        */
    }
    
    #[inline] pub fn get_axis(&self) -> i32 {
        
        todo!();
        /*
            return axis_;
        */
    }
    
    #[inline] pub fn get_add_axis(&self) -> bool {
        
        todo!();
        /*
            return addAxis_;
        */
    }
    
    #[inline] pub fn set_axis(&mut self, axis: i32)  {
        
        todo!();
        /*
            axis_ = axis;
        */
    }
    
    #[inline] pub fn set_add_axis(&mut self, add_axis: bool)  {
        
        todo!();
        /*
            addAxis_ = addAxis;
        */
    }
}

///-------------------------
pub struct Softmax {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Softmax);
    base: NeuralNetOperator,
}

impl Default for Softmax {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Softmax
        */
    }
}

///-------------------
pub struct ChannelShuffle {
    //NOMNIGRAPH_DEFINE_NN_RTTI(ChannelShuffle);
    base: NeuralNetOperator,
}

impl Default for ChannelShuffle {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::ChannelShuffle
        */
    }
}

///----------------------
pub struct Add {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Add);
    base: NeuralNetOperator,
    broadcast: i32,
}

impl Add {
    
    pub fn new(broadcast: Option<i32>) -> Self {
    
        let broadcast: i32 = broadcast.unwrap_or(0);

        todo!();
        /*
            : NeuralNetOperator(NNKind::Add), broadcast_(broadcast)
        */
    }
    
    #[inline] pub fn get_broadcast(&self) -> i32 {
        
        todo!();
        /*
            return broadcast_;
        */
    }
    
    #[inline] pub fn set_broadcast(&mut self, broadcast: i32)  {
        
        todo!();
        /*
            broadcast_ = broadcast;
        */
    }
}

///-------------------------------
pub struct Reshape {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Reshape);
    base: NeuralNetOperator,
}

impl Default for Reshape {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Reshape
        */
    }
}

///------------------
pub struct Flatten {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Flatten);
    base: NeuralNetOperator,
}

impl Default for Flatten {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Flatten
        */
    }
}

///--------------
pub struct CopyToOpenCL {
    //NOMNIGRAPH_DEFINE_NN_RTTI(CopyToOpenCL);
    base: NeuralNetOperator,
}

impl Default for CopyToOpenCL {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::CopyToOpenCL
        */
    }
}

///-----------------------
pub struct CopyFromOpenCL {
    //NOMNIGRAPH_DEFINE_NN_RTTI(CopyFromOpenCL);
    base: NeuralNetOperator,
}

impl Default for CopyFromOpenCL {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::CopyFromOpenCL
        */
    }
}

///---------------
pub struct NCHW2NHWC {
    //NOMNIGRAPH_DEFINE_NN_RTTI(NCHW2NHWC);
    base: NeuralNetOperator,
}

impl Default for NCHW2NHWC {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::NCHW2NHWC
        */
    }
}

///-----------------------
pub struct NHWC2NCHW {
    //NOMNIGRAPH_DEFINE_NN_RTTI(NHWC2NCHW);
    base: NeuralNetOperator,
}

impl Default for NHWC2NCHW {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::NHWC2NCHW
        */
    }
}

///----------------------
pub struct Declare {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Declare);
    base: NeuralNetOperator,
}

impl Default for Declare {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Declare
        */
    }
}

///----------------
pub struct Export {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Export);
    base: NeuralNetOperator,
}

impl Default for Export {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Export
        */
    }
}

