crate::ix!();

use crate::{
    OperatorStorage,
};

/**
  | FeedBlobs the content of the blobs.
  | The input and output blobs should be
  | one-to-one inplace.
  |
  */
pub struct FeedBlobOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    value:   String,
}

impl<Context> FeedBlobOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        CAFFE_ENFORCE(
            this->template HasSingleArgumentOfType<string>("value"),
            "value argument must exist and be passed as a string");
        value_ = this->template GetSingleArgument<string>("value", "");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *this->template Output<std::string>(0) = value_;
        return true;
        */
    }
}

register_cpu_operator!{FeedBlob, FeedBlobOp<CPUContext>}

should_not_do_gradient!{FeedBlob}

num_inputs!{FeedBlob, (0,0)}

num_outputs!{FeedBlob, (1,1)}

args!{FeedBlob, 
    0 => ("value", "(string) if provided then 
        we will use this string as the value for the provided output tensor")
}
