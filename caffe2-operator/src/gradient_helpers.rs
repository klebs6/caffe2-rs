crate::ix!();
   
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


