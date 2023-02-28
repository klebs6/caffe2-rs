crate::ix!();

/**
  | Computes Non-Maximum Suppression
  | on the GPU
  | 
  | Reject a bounding box if its region has
  | an intersection-overunion (IoU) overlap
  | with a higher scoring selected bounding
  | box larger than a threshold.
  | 
  | d_desc_sorted_boxes : pixel coordinates
  | of proposed bounding boxes size: (N,4),
  | format: [x1; y1; x2; y2] the boxes are
  | sorted by scores in descending order
  | 
  | N : number of boxes
  | 
  | d_keep_sorted_list : row indices of
  | the selected proposals, sorted by score
  | 
  | h_nkeep : number of selected proposals
  | 
  | dev_delete_mask, host_delete_mask
  | : Tensors that will be used as temp storage
  | by NMS
  | 
  | Those tensors will be resized to the
  | necessary size
  | 
  | context : current CUDA context
  |
  */
#[inline] pub fn nms_gpu_upright(
    d_desc_sorted_boxes: *const f32,
    n:                   i32,
    thresh:              f32,
    legacy_plus_one:     bool,
    d_keep_sorted_list:  *mut i32,
    h_nkeep:             *mut i32,
    dev_delete_mask:     &mut TensorCUDA,
    host_delete_mask:    &mut TensorCPU,
    context:             *mut CUDAContext)  {
    
    todo!();
    /*
    
    */
}

pub struct RotatedBox {
    x_ctr: f32,
    y_ctr: f32,
    w:     f32,
    h:     f32,
    a:     f32,
}

/**
  | Same as nms_gpu_upright, but for rotated
  | boxes with angle info.
  | 
  | d_desc_sorted_boxes : pixel coordinates
  | of proposed bounding boxes
  | 
  | size: (N,5), format: [x_ct; y_ctr;
  | width; height; angle]
  | 
  | the boxes are sorted by scores in descending
  | order
  |
  */
#[inline] pub fn nms_gpu_rotated(
    d_desc_sorted_boxes: *const f32,
    n:                   i32,
    thresh:              f32,
    d_keep_sorted_list:  *mut i32,
    h_nkeep:             *mut i32,
    dev_delete_mask:     &mut TensorCUDA,
    host_delete_mask:    &mut TensorCPU,
    context:             *mut CUDAContext)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn nms_gpu(
    d_desc_sorted_boxes: *const f32,
    n:                   i32,
    thresh:              f32,
    legacy_plus_one:     bool,
    d_keep_sorted_list:  *mut i32,
    h_nkeep:             *mut i32,
    dev_delete_mask:     &mut TensorCUDA,
    host_delete_mask:    &mut TensorCPU,
    context:             *mut CUDAContext,
    box_dim:             i32)  {
    
    todo!();
    /*
    
    */
}
