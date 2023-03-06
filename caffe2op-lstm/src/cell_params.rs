crate::ix!();

#[derive(Clone)]
pub struct CellParams {
    w_ih:    Tensor,
    w_hh:    Tensor,
    b_ih:    Option<Tensor>,
    b_hh:    Option<Tensor>,
    context: *mut CPUContext,
}

impl CellParams {
    
    pub fn new(
        _w_ih: &Tensor,
        _w_hh: &Tensor,
        _b_ih: &Tensor,
        _b_hh: &Tensor,
        _context: *mut CPUContext) -> Self 
    {
        todo!();
        /*
            initParams(_w_ih, _w_hh, _b_ih, _b_hh, _context);
        */
    }
    
    #[inline] pub fn init_params(
        &mut self, 
        _w_ih: &Tensor,
        _w_hh: &Tensor,
        _b_ih: &Tensor,
        _b_hh: &Tensor,
        _context: *mut CPUContext)
    {
        todo!();
        /*
            w_ih = copy_ctor(_w_ih);
        w_hh = copy_ctor(_w_hh);
        b_ih = copy_ctor(_b_ih);
        b_hh = copy_ctor(_b_hh);
        context = _context;
        */
    }
    
    #[inline] pub fn linear_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return linear(input, w_ih, b_ih, context);
        */
    }
    
    #[inline] pub fn linear_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return linear(h, w_hh, b_hh, context);
        */
    }
}
