/*!
  | A little explanation about why this file exists
  | at all.
  |
  | We have a few methods on Tensor class which
  | require access to reified access to
  | AutogradMeta.
  |
  | In open source, this isn't a big deal: we just
  | access torch/csrc/autograd/variable.h from
  | aten/src/ATen/core/Tensor.cpp and we can put
  | the definitions inline.
  |
  | This is because everything gets balled into
  | a single dynamic library in the end.
  |
  | However, inside our Facebook internal version
  | of our build system, we have a split between
  | aten and torch/csrc.  So we cannot simply just
  | cross this boundary.
  |
  | "Now wait," you might say, "Why don't we just
  | merge the libraries inside Facebook".
  |
  | Well, the problem is that there are some
  | downstream applications which are at binary
  | size limit, and incorporating all of the extra
  | code from libtorch would push them over
  | (admarket/adreview/service:adreviewservice, see
  | also
  | https://github.com/pytorch/pytorch/pull/29299)
  |
  | So if you want to do that, we have to fix all
  | of the services like this.
  |
  | I didn't want to block eliminating
  | Tensor-Variable on this work, so I had to
  | introduce another dynamic dispatch to get to
  | the variable implementations (which live in
  | torch/csrc/autograd/variable.cpp, FYI).
  |
  | I also considered using our existing dynamic
  | dispatch mechanism, c10 dispatcher, to do this.
  |
  | However, (1) some of the functions on Tensor
  | have weird signatures that are not supported by
  | autograd, and (2) see this bug
  | https://github.com/pytorch/pytorch/issues/30102
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/VariableHooksInterface.h]

pub trait VariableHooksInterface:
TensorData
+ VariableData
+ GradFn
+ RegisterHook
+ RemoveHook
+ IsView
+ Base
+ Name
+ IsLeaf
+ OutputNr
+ SetData
+ Data
+ Version
+ RetainGrad
+ RetainsGrad
+ Backward
+ RequiresGrad {}

pub trait TensorData {

    fn tensor_data(&self, _0: &Tensor) -> Tensor;
}

pub trait VariableData {
    
    fn variable_data(&self, _0: &Tensor) -> Tensor;
}

pub trait GradFn {
    
    fn grad_fn(&self, _0: &Tensor) -> &Arc<TorchautogradNode>;
}

pub trait RegisterHook {
    
    fn register_hook(&self, 
        _0:   &Tensor,
        hook: fn(_0: &Tensor) -> Tensor) -> u32;
}

pub trait RemoveHook {
    
    fn remove_hook(&self, 
        _0:  &Tensor,
        pos: u32);
}

pub trait IsView {
    
    fn is_view(&self, _0: &Tensor) -> bool;
}

pub trait Base {
    
    fn base(&self, _0: &Tensor) -> &Tensor;
}

pub trait Name {

    fn name(&self, _0: &Tensor) -> &String;
}

pub trait IsLeaf {
    
    fn is_leaf(&self, _0: &Tensor) -> bool;
}

pub trait OutputNr {
    
    fn output_nr(&self, _0: &Tensor) -> i64;
}

pub trait SetData {
    
    fn set_data(&self, 
        _0: &Tensor,
        _1: &Tensor);
}

pub trait Data {
    
    fn data(&self, _0: &Tensor) -> Tensor;
}

pub trait Version {
    
    fn version(&self, _0: &Tensor) -> i64;
}

pub trait RetainGrad {
    
    fn retain_grad(&self, _0: &Tensor);
}

pub trait RetainsGrad {
    
    fn retains_grad(&self, _0: &Tensor) -> bool;
}

pub trait Backward {
    
    fn backward(&self, 
        _0: &Tensor,
        _1: TensorList,
        _2: &Option<Tensor>,
        _3: Option<bool>,
        _4: bool);
}

pub trait RequiresGrad {
    
    fn requires_grad(&self, 
        _0: &Tensor,
        _1: bool);
}

pub struct VariableHooksRegisterer {

}

impl VariableHooksRegisterer {
    
    pub fn new(hooks: *mut VariableHooksInterface) -> Self {
    
        todo!();
        /*


            SetVariableHooks(hooks);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/VariableHooksInterface.cpp]

lazy_static!{
    /*
    VariableHooksInterface* hooks = nullptr;
    */
}

pub fn set_variable_hooks(h: *mut VariableHooksInterface)  {
    
    todo!();
        /*
            hooks = h;
        */
}

pub fn get_variable_hooks() -> *mut VariableHooksInterface {
    
    todo!();
        /*
            TORCH_CHECK(hooks, "Support for autograd has not been loaded; have you linked against libtorch.so?")
      return hooks;
        */
}
