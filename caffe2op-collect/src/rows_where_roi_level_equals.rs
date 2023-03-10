crate::ix!();

/**
  | Update out_filtered and out_indices
  | with rows from rois where lvl matches
  | value in lvls passed in.
  |
  */
#[inline] pub fn rows_where_roi_level_equals(
    rois:         &ERArrXXf,
    lvls:         &ERArrXXf,
    lvl:          i32,
    out_filtered: *mut ERArrXXf,
    out_indices:  *mut EArrXi)  {
    
    todo!();
    /*
        CAFFE_ENFORCE(out_filtered != nullptr, "Output filtered required");
      CAFFE_ENFORCE(out_indices != nullptr, "Output indices required");
      CAFFE_ENFORCE(rois.rows() == lvls.rows(), "RoIs and lvls count mismatch");
      // Calculate how many rows we need
      int filtered_size = (lvls == lvl).rowwise().any().count();
      // Fill in the rows and indices
      out_filtered->resize(filtered_size, rois.cols());
      out_indices->resize(filtered_size);
      for (int i = 0, filtered_idx = 0; i < rois.rows(); i++) {
        auto lvl_row = lvls.row(i);
        if ((lvl_row == lvl).any()) {
          out_filtered->row(filtered_idx) = rois.row(i);
          (*out_indices)(filtered_idx) = i;
          filtered_idx++;
        }
      }
    */
}
