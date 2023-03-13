crate::ix!();

pub struct GetEnsureDenseGradient;

impl GetGradientDefs for GetEnsureDenseGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            GradOut(0).IsSparse() || GradOut(0).IsDense(),
            "Input gradient ",
            O(0),
            " should be either sparse or dense.");

        if (GradOut(0).IsDense()) {
          SetDense(0, GO(0));
          return vector<OperatorDef>();
        } else {
          return SingleGradientDef(
              "SparseToDense",
              "",
              vector<string>{GO_I(0), GO_V(0), I(0)},
              vector<string>{GI(0)});
        }
        */
    }
}
