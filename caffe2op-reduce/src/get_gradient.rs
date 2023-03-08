crate::ix!();

pub struct GetReduceFrontMeanGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceFrontMeanGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceFrontMeanGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    ReduceFrontMean, 
    GetReduceFrontMeanGradient
}

pub struct GetReduceBackMeanGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceBackMeanGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceBackMeanGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    ReduceBackMean, 
    GetReduceBackMeanGradient
}

pub struct GetReduceFrontSumGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceFrontSumGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceFrontSumGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    ReduceFrontSum, 
    GetReduceFrontSumGradient
}

pub struct GetReduceBackSumGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceBackSumGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceBackSumGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    ReduceBackSum, 
    GetReduceBackSumGradient
}

pub struct GetReduceBackMaxGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceBackMaxGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0), O(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceBackMaxGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    ReduceBackMax, 
    GetReduceBackMaxGradient
}

pub struct GetReduceFrontMaxGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceFrontMaxGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0), O(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceFrontMaxGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReduceFrontMax, GetReduceFrontMaxGradient}

pub struct GetReduceGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<string>{GO(0), I(0), O(0)},
            std::vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReduceMin,  GetReduceGradient}
register_gradient!{ReduceMax,  GetReduceGradient}
register_gradient!{ReduceSum,  GetReduceGradient}
register_gradient!{ReduceMean, GetReduceGradient}
register_gradient!{ReduceL1,   GetReduceGradient}
register_gradient!{ReduceL2,   GetReduceGradient}
