crate::ix!();

pub trait AddRelatedBlobInfo {

    #[inline] fn add_related_blob_info(&mut self, err: *mut EnforceNotMet)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          isLegacyOperator(),
          "AddRelatedBlobInfo(err) not supported for operators exported to c10.");

      if (!has_debug_def()) {
        return;
      }

      bool found_input = false;
      bool found_output = false;
      if (err->caller() != nullptr) {
        std::ostringstream oss;
        for (size_t i = 0; i < inputs_.size(); i++) {
          if (inputs_[i]->GetRaw() == err->caller()) {
            found_input = true;
            oss << "while accessing input: " << debug_def().input(i);
            break;
          }
        }
        for (size_t i = 0; i < outputs_.size(); i++) {
          if (outputs_[i]->GetRaw() == err->caller()) {
            found_output = true;
            if (found_input) {
              oss << " OR ";
            }
            oss << "while accessing output: " << debug_def().output(i);
            break;
          }
        }
        if (found_input || found_output) {
          err->add_context(oss.str());
        }
      }
        */
    }
}
