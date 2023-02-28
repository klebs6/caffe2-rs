/*!
 | Utilities that make it easier to write caffe2 C++
 | unit tests.
 |
 | These utils are designed to be concise and easy to
 | use. They may sacrifice performance and should
 | only be used in tests/non production code.
 */

crate::ix!();

use crate::{
    OperatorDef,
    NetDef,
    Tensor,
    Workspace,
    TensorCPU,
};

/**
  | Asserts that the numeric values of a
  | tensor is equal to a data vector.
  |
  */
#[inline] pub fn assert_tensor_equals_data<T>(
    tensor:  &TensorCPU,
    data:    &Vec<T>,
    epsilon: f32) 
{
    todo!();
    /*
        CAFFE_ENFORCE(tensor.IsType<T>());
      CAFFE_ENFORCE_EQ(tensor.numel(), data.size());
      for (auto idx = 0; idx < tensor.numel(); ++idx) {
        if (tensor.IsType<float>()) {
          assertNear(tensor.data<T>()[idx], data[idx], epsilon);
        } else {
          CAFFE_ENFORCE_EQ(tensor.data<T>()[idx], data[idx]);
        }
      }
    */
}

/// Assertion for tensor sizes and values.
#[inline] pub fn assert_tensor<T>(
    tensor:  &TensorCPU,
    sizes:   &Vec<i64>,
    data:    &Vec<T>,
    epsilon: f32) 
{
    todo!();
    /*
        CAFFE_ENFORCE_EQ(tensor.sizes(), sizes);
      assertTensorEquals(tensor, data, epsilon);
    */
}

/**
  | Fill a buffer with randomly generated
  | numbers given range [min, max) T can
  | only be float, double or long double
  | default RealType = float
  |
  */
#[inline] pub fn random_fill<RealType>(
    data: *mut RealType,
    size: usize,
    min:  f64,
    max:  f64) 
{
    todo!();
    /*
        std::mt19937 gen(42);
      std::uniform_real_distribution<RealType> dis(
          static_cast<RealType>(min), static_cast<RealType>(max));
      for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
      }
    */
}

/// Fill data from a vector to a tensor.
#[inline] pub fn fill_tensor<T>(
    shape:  &Vec<i64>,
    data:   &Vec<T>,
    tensor: *mut TensorCPU) 
{
    todo!();
    /*
        tensor->Resize(shape);
      CAFFE_ENFORCE_EQ(data.size(), tensor->numel());
      auto ptr = tensor->mutable_data<T>();
      for (int i = 0; i < tensor->numel(); ++i) {
        ptr[i] = data[i];
      }
    */
}

/// Create a tensor and fill data.
#[inline] pub fn create_tensor_in_workspace_and_fill<T>(
    name:      &String,
    shape:     &Vec<i64>,
    data:      &Vec<T>,
    workspace: *mut Workspace) -> *mut Tensor 
{
    todo!();
    /*
        auto* tensor = createTensor(name, workspace);
      fillTensor<T>(shape, data, tensor);
      return tensor;
    */
}

#[inline] pub fn create_tensor_and_fill<T>(
    shape: &Vec<i64>,
    data:  &Vec<T>) -> Tensor 
{
    todo!();
    /*
        Tensor tensor(caffe2::CPU);
      fillTensor<T>(shape, data, &tensor);
      return tensor;
    */
}

/// Fill a constant to a tensor.
#[inline] pub fn constant_fill_tensor<T>(
    shape:  &Vec<i64>,
    data:   &T,
    tensor: *mut TensorCPU) 
{
    todo!();
    /*
        tensor->Resize(shape);
      auto ptr = tensor->mutable_data<T>();
      for (int i = 0; i < tensor->numel(); ++i) {
        ptr[i] = data;
      }
    */
}

/// Create a tensor and fill a constant.
#[inline] pub fn create_tensor_and_constant_fill<T>(
    name:      &String,
    shape:     &Vec<i64>,
    data:      &T,
    workspace: *mut Workspace) -> *mut Tensor 
{
    todo!();
    /*
        auto* tensor = createTensor(name, workspace);
      constantFillTensor<T>(shape, data, tensor);
      return tensor;
    */
}

/// Concise util class to mutate a net in a chaining fashion.
pub struct NetMutator {
    net:             *mut NetDef,
    last_created_op: *mut OperatorDef,
}

impl NetMutator {

    /// Add argument to the last created op.
    #[inline] pub fn add_argument<T>(
        &mut   self, 
        name:  &String,
        value: &T) -> &mut NetMutator 
    {
        todo!();
        /*
            CAFFE_ENFORCE(lastCreatedOp_ != nullptr);
        AddArgument(name, value, lastCreatedOp_);
        return *this;
        */
    }
    
    pub fn new(net: *mut NetDef) -> Self {
        todo!();
        /*
            : net_(net)
        */
    }
}

/**
  | Concise util class to mutate a workspace
  | in a chaining fashion.
  |
  */
pub struct WorkspaceMutator {
    workspace: *mut Workspace,
}

impl WorkspaceMutator {
    
    pub fn new(workspace: *mut Workspace) -> Self {
        todo!();
        /*
            : workspace_(workspace)
        */
    }

    /// New tensor filled by a data vector.
    #[inline] pub fn new_tensor<T>(
        &mut self, 
        name:  &String,
        shape: &Vec<i64>,
        data:  &Vec<T>) -> &mut WorkspaceMutator 
    {
        todo!();
        /*
            createTensorAndFill<T>(name, shape, data, workspace_);
        return *this;
        */
    }

    /// New tensor filled by a constant.
    #[inline] pub fn new_tensor_const<T>(
        &mut self, 
        name:  &String,
        shape: &Vec<i64>,
        data:  &T) -> &mut WorkspaceMutator 
    {
        todo!();
        /*
            createTensorAndConstantFill<T>(name, shape, data, workspace_);
        return *this;
        */
    }
}

#[inline] pub fn assert_tensor_equals_with_type<T>(
    tensor1: &TensorCPU,
    tensor2: &TensorCPU,
    unused:  f32) 
{
    todo!("dispatch");
    /*
        CAFFE_ENFORCE_EQ(tensor1.sizes(), tensor2.sizes());
      for (auto idx = 0; idx < tensor1.numel(); ++idx) {
        CAFFE_ENFORCE_EQ(tensor1.data<T>()[idx], tensor2.data<T>()[idx]);
      }
    */
}

#[inline] pub fn assert_tensor_equals_with_type_f32(
    tensor1: &TensorCPU,
    tensor2: &TensorCPU,
    eps:  f32) 
{
    todo!();
    /*
      CAFFE_ENFORCE_EQ(tensor1.sizes(), tensor2.sizes());
      for (auto idx = 0; idx < tensor1.numel(); ++idx) {
        CAFFE_ENFORCE_LT(
            fabs(tensor1.data<float>()[idx] - tensor2.data<float>()[idx]),
            eps,
            "Mismatch at index ",
            idx,
            " exceeds threshold of ",
            eps);
      }
    */
}

/**
  | Asserts that two float values are close
  | within epsilon.
  |
  */
#[inline] pub fn assert_near(
    value1:  f32,
    value2:  f32,
    epsilon: f32) 
{
    todo!();
    /*
        // These two enforces will give good debug messages.
      CAFFE_ENFORCE_LE(value1, value2 + epsilon);
      CAFFE_ENFORCE_GE(value1, value2 - epsilon);
    */
}

/**
  | Asserts that the values of two tensors
  | are the same.
  |
  */
#[inline] pub fn assert_tensor_equals(
    tensor1: &TensorCPU,
    tensor2: &TensorCPU,
    eps:     Option<f32>)
{
    let eps = eps.unwrap_or(1e-6);
    
    todo!();
    /*
        CAFFE_ENFORCE_EQ(tensor1.sizes(), tensor2.sizes());
      if (tensor1.IsType<float>()) {
        CAFFE_ENFORCE(tensor2.IsType<float>());
        assertTensorEqualsWithType<float>(tensor1, tensor2, eps);
      } else if (tensor1.IsType<int>()) {
        CAFFE_ENFORCE(tensor2.IsType<int>());
        assertTensorEqualsWithType<int>(tensor1, tensor2, eps);
      } else if (tensor1.IsType<int64_t>()) {
        CAFFE_ENFORCE(tensor2.IsType<int64_t>());
        assertTensorEqualsWithType<int64_t>(tensor1, tensor2, eps);
      }
      // Add more types if needed.
    */
}

/**
  | Asserts a list of tensors presented
  | in two workspaces are equal.
  |
  */
#[inline] pub fn assert_tensor_list_equals(
    tensor_names: &Vec<String>,
    workspace1: &Workspace,
    workspace2: &Workspace)
{
    todo!();
    /*
        for (const std::string& tensorName : tensorNames) {
        CAFFE_ENFORCE(workspace1.HasBlob(tensorName));
        CAFFE_ENFORCE(workspace2.HasBlob(tensorName));
        auto& tensor1 = getTensor(workspace1, tensorName);
        auto& tensor2 = getTensor(workspace2, tensorName);
        assertTensorEquals(tensor1, tensor2);
      }
    */
}

/// Read a tensor from the workspace.
#[inline] pub fn get_tensor<'a>(
    workspace: &'a Workspace,
    name:      &'a String) -> &'a Tensor 
{
    todo!();
    /*
        CAFFE_ENFORCE(workspace.HasBlob(name));
      return workspace.GetBlob(name)->Get<caffe2::Tensor>();
    */
}

/// Create a new tensor in the workspace.
#[inline] pub fn create_tensor(
    name: &String,
    workspace: *mut Workspace) -> *mut Tensor 
{
    todo!();
    /*
        return BlobGetMutableTensor(workspace->CreateBlob(name), caffe2::CPU);
    */
}

/// Create a new operator in the net.
#[inline] pub fn create_operator(
    ty:      &String,
    inputs:  &Vec<String>,
    outputs: &Vec<String>,
    net:     *mut NetDef) -> *mut OperatorDef 
{
    todo!();
    /*
        auto* op = net->add_op();
      op->set_type(type);
      for (const auto& in : inputs) {
        op->add_input(in);
      }
      for (const auto& out : outputs) {
        op->add_output(out);
      }
      return op;
    */
}

impl NetMutator {
    
    #[inline] pub fn new_op(
        &mut self, 
        ty:      &String,
        inputs:  &Vec<String>,
        outputs: &Vec<String>) -> &mut NetMutator {
        
        todo!();
        /*
            lastCreatedOp_ = createOperator(type, inputs, outputs, net_);
      return *this;
        */
    }
    
    #[inline] pub fn external_inputs(&mut self, external_inputs: &Vec<String>) -> &mut NetMutator {
        
        todo!();
        /*
            for (auto& blob : externalInputs) {
        net_->add_external_input(blob);
      }
      return *this;
        */
    }
    
    #[inline] pub fn external_outputs(&mut self, external_outputs: &Vec<String>) -> &mut NetMutator {
        
        todo!();
        /*
            for (auto& blob : externalOutputs) {
        net_->add_external_output(blob);
      }
      return *this;
        */
    }
    
    /// Set device name for the last created op.
    #[inline] pub fn set_device_option_name(&mut self, name: &String) -> &mut NetMutator {
        
        todo!();
        /*
            CAFFE_ENFORCE(lastCreatedOp_ != nullptr);
      lastCreatedOp_->mutable_device_option()->set_node_name(name);
      return *this;
        */
    }
}
