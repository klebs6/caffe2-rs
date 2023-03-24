crate::ix!();

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
 
