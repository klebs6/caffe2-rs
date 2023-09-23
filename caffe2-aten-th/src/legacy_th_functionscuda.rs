crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/LegacyTHFunctionsCUDA.h]

pub fn th_masked_fill<'a>(
        self_: &mut Tensor,
        mask:  &Tensor,
        value: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn th_masked_fill_bool<'a>(
        self_: &mut Tensor,
        mask:  &Tensor,
        value: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn th_cross_kernel_out<'a>(
        result: &mut Tensor,
        self_:  &Tensor,
        other:  &Tensor,
        dim:    i64) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn th_cross_kernel(
        self_: &Tensor,
        other: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn th_gels_out<'a>(
        self_: &Tensor,
        A:     &Tensor,
        res1:  &mut Tensor,
        res2:  &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn th_gels(
        self_: &Tensor,
        A:     &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn th_potri_out<'a>(
        output: &mut Tensor,
        self_:  &Tensor,
        upper:  bool) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn th_potri(
        self_: &Tensor,
        upper: bool) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn th_copy_ignoring_overlaps<'a>(
        self_: &mut Tensor,
        src:   &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_multi_margin_loss_forward_out<'a>(
        self_:      &Tensor,
        target:     &Tensor,
        p:          &Scalar,
        margin:     &Scalar,
        weight_opt: &Option<Tensor>,
        reduction:  i64,
        output:     &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_multi_margin_loss_forward(
        self_:     &Tensor,
        target:    &Tensor,
        p:         &Scalar,
        margin:    &Scalar,
        weight:    &Option<Tensor>,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_multi_margin_loss_backward_out<'a>(
        grad_output: &Tensor,
        self_:       &Tensor,
        target:      &Tensor,
        p:           &Scalar,
        margin:      &Scalar,
        weight_opt:  &Option<Tensor>,
        reduction:   i64,
        grad_input:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_multi_margin_loss_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        target:      &Tensor,
        p:           &Scalar,
        margin:      &Scalar,
        weight:      &Option<Tensor>,
        reduction:   i64) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_multilabel_margin_loss_forward_out<'a>(
        self_:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        output:    &mut Tensor,
        is_target: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_multilabel_margin_loss_forward(
        self_:     &Tensor,
        target:    &Tensor,
        reduction: i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_multilabel_margin_loss_backward_out<'a>(
        grad_output: &Tensor,
        self_:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        is_target:   &Tensor,
        grad_input:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_multilabel_margin_loss_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        is_target:   &Tensor) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_nll_loss_forward_out<'a>(
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        output:       &mut Tensor,
        total_weight: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_nll_loss_forward(
        self_:        &Tensor,
        target:       &Tensor,
        weight:       &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_nll_loss_backward_out<'a>(
        grad_output:  &Tensor,
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor,
        grad_input:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_nll_loss_backward(
        grad_output:  &Tensor,
        self_:        &Tensor,
        target:       &Tensor,
        weight:       &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_nll_loss2d_forward_out<'a>(
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        output:       &mut Tensor,
        total_weight: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_nll_loss2d_forward(
        self_:        &Tensor,
        target:       &Tensor,
        weight:       &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_nll_loss2d_backward_out<'a>(
        grad_output:  &Tensor,
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor,
        grad_input:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_nll_loss2d_backward(
        grad_output:  &Tensor,
        self_:        &Tensor,
        target:       &Tensor,
        weight:       &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_glu_forward_out<'a>(
        self_:  &Tensor,
        dim:    i64,
        output: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_glu_forward(
        self_: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_glu_backward_out<'a>(
        grad_output: &Tensor,
        self_:       &Tensor,
        dim:         i64,
        grad_input:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_glu_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        dim:         i64) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_log_sigmoid_forward_out<'a>(
        self_:  &Tensor,
        output: &mut Tensor,
        buffer: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_log_sigmoid_forward(self_: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_log_sigmoid_backward_out<'a>(
        grad_output: &Tensor,
        self_:       &Tensor,
        buffer:      &Tensor,
        grad_input:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_log_sigmoid_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        buffer:      &Tensor) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_rrelu_with_noise_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        noise:       &Tensor,
        lower:       &Scalar,
        upper:       &Scalar,
        training:    bool) -> Tensor {
    
    todo!();
        /*
        
        */
}


pub fn thnn_conv2d_forward_out<'a>(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        output:      &mut Tensor,
        columns:     &mut Tensor,
        ones:        &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_conv2d_forward(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias:        &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}


pub fn thnn_conv2d_backward_out<'a>(
        grad_input:  &mut Tensor,
        grad_weight: &mut Tensor,
        grad_bias:   &mut Tensor,
        grad_output: &Tensor,
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        columns:     &Tensor,
        ones:        &Tensor) -> (&'a mut Tensor,&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
        
        */
}

pub fn thnn_conv2d_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        columns:     &Tensor,
        ones:        &Tensor,
        output_mask: BoolArray3) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}

pub fn thnn_conv_depthwise2d_forward_out<'a>(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        output:      &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}

pub fn thnn_conv_depthwise2d_forward(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias:        &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32]) -> Tensor {
    
    todo!();
        /*
        
        */
}

pub fn thnn_conv_depthwise2d_backward_out<'a>(
        grad_input:  &mut Tensor,
        grad_weight: &mut Tensor,
        grad_output: &Tensor,
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32]) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
        
        */
}

pub fn thnn_conv_depthwise2d_backward(
        grad_output: &Tensor,
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        output_mask: BoolArray2) -> (Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}
