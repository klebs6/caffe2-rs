crate::ix!();

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
