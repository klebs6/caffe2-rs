crate::ix!();

impl<T,Context> Drop for SummarizeOp<T,Context> {

    fn drop(&mut self) {
        todo!();
        /* 
        if (to_file_)
          log_file_->close();
       */
    }
}

impl<T,Context> SummarizeOp<T,Context> {

    const MIN_IDX:   i32 = 0;
    const MAX_IDX:   i32 = 1;
    const MEAN_IDX:  i32 = 2;
    const STD_IDX:   i32 = 3;
    const NUM_STATS: i32 = 4;

    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(def, ws),
            to_file_(this->template GetSingleArgument<int>("to_file", 0)) 
        if (to_file_) {
          // We will output to file instead of printing on screen.
          const string& target_folder = ws->RootFolder();
          // We will write each individual tensor to its individual file.
          // Also, since the namescope is currently represented by "/", we will
          // need to replace it with a symbol that does not conflict with the
          // folder separator in Linux.
          string proper_name = def.input(0);
          std::replace(proper_name.begin(), proper_name.end(), '/', '#');
          log_file_.reset(new std::ofstream(
              target_folder + "/" + proper_name + kSummaryzeOpExtension,
              std::ofstream::out | std::ofstream::trunc));
          CAFFE_ENFORCE(
              log_file_->good(),
              "Failed to open summarize file for tensor ",
              def.input(0),
              ". rdstate() = ",
              log_file_->rdstate());
        }
        */
    }
}
