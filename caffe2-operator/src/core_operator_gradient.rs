crate::ix!();

/**
  | -----------
  | @brief
  | 
  | A struct that abstracts on top of dense
  | and sparse blobs.
  | 
  | For a dense blob, its gradient name should
  | be written into dense_, and for a sparse
  | blob, its gradient name should be written
  | into indice_ for the sparse indices
  | and value_ for the values.
  |
  */
pub struct GradientWrapper {
    dense:   String,
    indices: String,
    values:  String,
}

impl GradientWrapper {

    #[inline] pub fn is_dense() -> bool {
        
        todo!();
        /*
            return (dense_.size() != 0);
        */
    }


    #[inline] pub fn is_sparse() -> bool {
        
        todo!();
        /*
            return (indices_.size() != 0 || values_.size() != 0);
        */
    }


    #[inline] pub fn is_empty() -> bool {
        
        todo!();
        /*
            return (!IsDense() && !IsSparse());
        */
    }
}

/**
  | A struct that holds the gradient operators
  | and related gradient maps.
  |
  */
#[derive(Default)]
pub struct GradientOpsMeta {
    ops:     Vec<OperatorDef>,
    g_input: Vec<GradientWrapper>,
}

pub struct GradientMakerStorage<'a> {
    def:      &'a OperatorDef,
    g_output: &'a Vec<GradientWrapper>,
    g_input:  Vec<GradientWrapper>,
}

impl<'a> GradientMakerStorage<'a> {
    
    pub fn new(
        def:      &OperatorDef, 
        g_output: &Vec<GradientWrapper>) -> Self {

        todo!();
        /*
            : def_(def), g_output_(g_output), g_input_(def.input_size())
        */
    }
}

pub trait GradientMakerBase = 
CopyDeviceOption 
+ CopyArguments 
+ CopyEngine 
+ GetGradientDefs
+ GetGradientOpsMeta 
+ GetOperatorDef 
+ GradOut 
+ GradientHelpers 
+ GradientName 
+ GradientNameToParam 
+ GradientSliceIndices 
+ GradientSliceValues 
+ IsGradientBlob 
+ MatchGradsToParams 
+ SetDense 
+ SetSparse 
+ SingleGradientDef 
+ VerifyOp 
;

pub trait CopyDeviceOption {
    
    #[inline] fn copy_device_option(&self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}
    
pub trait CopyEngine {

    #[inline] fn copy_engine(&self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}
    
pub trait VerifyOp {
    
    #[inline] fn verify_op(&self)  {
        
        todo!();
        /*
            auto* schema = OpSchemaRegistry::Schema(def_.type());
        if (schema) {
          CAFFE_ENFORCE(
              schema->Verify(def_),
              "(GradientMaker) Operator def did not pass schema checking: ",
              ProtoDebugString(def_));
        }
        */
    }
}
    
pub trait GetGradientOpsMeta {

    /**
      | -----------
      | @brief
      | 
      | Returns the gradient ops meta.
      | 
      | If your gradient op generator only use
      | standard input and output manipulations,
      | you can simply implement GetGradientDefs()
      | that returns vector<OperatorDef>.
      | 
      | In that, you can call GI, GI_V and GI_I
      | that will automatically create the
      | gradient registration for you.
      | 
      | If you need to do custom gradient name
      | registration, overload this function
      | directly.
      |
      */
    #[inline] fn get(&mut self) -> GradientOpsMeta {
        
        todo!();
        /*
            VerifyOp();
        vector<OperatorDef> new_defs = GetGradientDefs();
        for (auto& opdef : new_defs) {
          opdef.set_is_gradient_op(true);
        }
        return GradientOpsMeta(new_defs, g_input_);
        */
    }
}

pub trait GetOperatorDef {
    
    #[inline] fn def(&self) -> &OperatorDef {
        
        todo!();
        /*
            return def_;
        */
    }
}
    
pub trait GradientHelpers {

    /**
      | Helper functions to return names for the
      | gradient computation.
      |
      | I(idx), O(idx): return the input and
      | output names.
      |
      | GO(idx): return the name of the gradient
      | for output idx.
      |
      | GI(idx), GI_I(idx), GI_V(idx): return the
      |     name of the gradient for input idx,
      |     and also registers that name into the
      |     gradient registry to be returned.
      */
    #[inline] fn i(&mut self, i: i32) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE((i >= 0) && (i < def_.input().size()));
        return def_.input(i);
        */
    }
    
    #[inline] fn o(&mut self, i: i32) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE((i >= 0) && (i < def_.output().size()));
        return def_.output(i);
        */
    }
    
    #[inline] fn gI(&mut self, i: i32) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            !g_input_.at(i).IsSparse(),
            "Input ",
            def_.input(i),
            " already set to sparse.");
        g_input_.at(i).dense_ = GradientName(def_.input(i));
        return GradientName(def_.input(i));
        */
    }
    
    #[inline] fn gI_I(&mut self, i: i32) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            !g_input_.at(i).IsDense(),
            "Input ",
            def_.input(i),
            " already set to dense.");
        g_input_.at(i).indices_ = GradientSliceIndices(def_.input(i));
        return GradientSliceIndices(def_.input(i));
        */
    }
    
    #[inline] fn gI_V(&mut self, i: i32) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            !g_input_.at(i).IsDense(),
            "Input ",
            def_.input(i),
            " already set to dense.");
        g_input_.at(i).values_ = GradientSliceValues(def_.input(i));
        return GradientSliceValues(def_.input(i));
        */
    }
    
    #[inline] fn gO(&mut self, i: i32) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            g_output_.at(i).IsDense(),
            "Gradient of output ",
            def_.output(i),
            (g_output_.at(i).IsSparse() ? " is sparse (expected dense)."
                                        : " is not provided!"));
        return g_output_.at(i).dense_;
        */
    }
    
    #[inline] fn gO_I(&mut self, i: i32) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            g_output_.at(i).IsSparse(),
            "Gradient of output ",
            def_.output(i),
            (g_output_.at(i).IsDense() ? " is dense (expected sparse)."
                                       : " is not provided!"));
        return g_output_.at(i).indices_;
        */
    }
    
    #[inline] fn gO_V(&mut self, i: i32) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            g_output_.at(i).IsSparse(),
            "Gradient of output ",
            def_.output(i),
            (g_output_.at(i).IsDense() ? " is dense (expected sparse)."
                                       : " is not provided!"));
        return g_output_.at(i).values_;
        */
    }
}

pub trait GradOut {

    fn grad_out(&mut self, i: i32) -> &GradientWrapper {
        
        todo!();
        /*
            return g_output_.at(i);
        */
    }
}

pub trait SetDense {

    /// Function to add a gradient pair to map.
    fn set_dense(
        &mut self, 
        i: i32,
        name: &String)
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            !g_input_.at(i).IsSparse(),
            "Input ",
            def_.input(i),
            " already set to sparse.");
        g_input_.at(i).dense_ = name;
        */
    }
}

pub trait SetSparse {

    #[inline] fn set_sparse(
        &mut self, 
        i: i32,
        indices: &String,
        values: &String)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            !g_input_.at(i).IsDense(),
            "Input ",
            def_.input(i),
            " already set to dense.");
        g_input_.at(i).indices_ = indices;
        g_input_.at(i).values_ = values;
        */
    }
}

pub trait SingleGradientDef {

    /**
      | -----------
      | @brief
      | 
      | a helper function to allow one to create
      | one single operator def, which is usually
      | the case for many simple operators.
      |
      */
    fn single_gradient_def<Args>(args: &Args) -> Vec<OperatorDef> {
        todo!();
        /*
            return vector<OperatorDef>{CreateOperatorDef(args...)};
        */
    }
}

pub trait MatchGradsToParams {

    /**
      | Returns map that returns the parameters
      | that the gradients are for.
      |
      */
    fn match_grads_to_params(op: &OperatorDef) -> HashMap<String,String> {
        
        todo!();
        /*
            // NOTE: how to go beyond string-matching?
        CaffeMap<string, string> m;
        for (auto& out : op.output()) {
          if (IsGradientBlob(out)) {
            m[out] = out.substr(0, out.length() - 5);
          }
        }
        return m;
        */
    }
}

pub trait GradientName {

    /**
      | Utility functions for gradient name
      | computation. We don't expose them in order
      | to discourage the use of such names
      | explicitly.
      */
    fn gradient_name(name: &String) -> String {
        
        todo!();
        /*
            return name + "_grad";
        */
    }
}

pub trait IsGradientBlob {

    fn is_gradient_blob(name: &String) -> bool {
        
        todo!();
        /*
            return name.length() > 5 && name.find("_grad") == name.length() - 5;
        */
    }
}

pub trait GradientNameToParam {

    fn gradient_name_to_param(name: &String) -> String {
        
        todo!();
        /*
            CHECK(IsGradientBlob(name));
        return name.substr(0, name.length() - 5);
        */
    }
}

pub trait GradientSliceIndices {

    fn gradient_slice_indices(name: &String) -> String {
        
        todo!();
        /*
            return name + "_grad_indices";
        */
    }
}

pub trait GradientSliceValues {

    fn gradient_slice_values(name: &String) -> String {
        
        todo!();
        /*
            return name + "_grad_values";
        */
    }
}

pub trait CopyArguments {

    #[inline] fn copy_arguments(&self) -> bool {
        true
    }
}

pub trait GetGradientDefs {
    fn get_gradient_defs(&mut self) -> Vec<OperatorDef>;
}

/**
  | -----------
  | @brief
  | 
  | A helper class to indicate that the operator
  | does not need gradient computation.
  | 
  | Use the macro NO_GRADIENT to register
  | operators that do not have gradients.
  | 
  | -----------
  | @note
  | 
  | this is different fron SHOULD_NOT_DO_GRADIENT:
  | the latter means that the gradient computation
  | should not flow through it at all, and
  | throws an error if it is called.
  |
  */
pub struct oGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for oGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return vector<OperatorDef>();
        */
    }
}

/**
  | -----------
  | @brief
  | 
  | A helper class to indicate that the operator
  | should have no gradient.
  | 
  | This is used when the operator definition
  | is designed to not have a gradient.
  | 
  | Calling a gradient on this operator
  | def will cause Caffe2 to quit.
  |
  */
pub struct ThrowInTheTowelIfGradientIsCalled<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> ThrowInTheTowelIfGradientIsCalled<'a> {
    
    #[inline] pub fn get(&mut self) -> GradientOpsMeta {
        
        todo!();
        /*
            CAFFE_THROW("One should not call gradient for operator ", def_.type(), ".");
        */
    }
}

/**
  | -----------
  | @brief
  | 
  | A helper class to indicate that the gradient
  | mechanism is not ready.
  | 
  | This should only be used sparsely when
  | the gradient does exist, but we have
  | not implemented it yet and are using
  | this as a lazy excuse. Eventually, a
  | gradient operator should be implemented.
  |
  */
pub struct GradientNotImplementedYet<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GradientNotImplementedYet<'a> {
    
    #[inline] pub fn get(&mut self) -> GradientOpsMeta {
        
        todo!();
        /*
            CAFFE_THROW(
            "Operator ",
            def_.type(),
            " should have a gradient but is not implemented yet.");
        */
    }
}

declare_registry!{
    GradientRegistry,
    GradientMakerBase,
    OperatorDef,
    Vec<GradientWrapper>
}

/**
  | -----------
  | @brief
  | 
  | Gets the GradientOpsMeta for the given
  | operator def.
  |
  */
#[inline] pub fn get_gradient_for_op(
    def: &OperatorDef, 
    g_output: &Vec<GradientWrapper>) -> GradientOpsMeta 
{
    todo!();
    /*
    
    */
}
