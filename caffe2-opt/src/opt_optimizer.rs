crate::ix!();

use crate::{
    NNModule,
    Workspace,
    NetDef
};

#[inline] pub fn workspace_optimizations<T,U>(
    nn:    *mut NNModule<T,U>,
    ws:    *mut Workspace,
    level: i32)  {
    
    todo!();
    /*
        switch (level) {
        case 1:
          opt::fuseConvBN(nn, ws);
        case 0:
        default:
          break;
      }
    */
}

#[inline] pub fn graph_optimzations<T,U>(nn: *mut NNModule<T,U>, level: i32)  {
    
    todo!();
    /*
        switch (level) {
        case 1:
    #ifdef USE_NNPACK
          opt::addNNPACK(nn, false);
          opt::fuseNNPACKConvRelu(nn);
    #endif
        case 0:
        default:
          break;
      }
    */
}

#[inline] pub fn optimize_with_workspace(
    net:   NetDef, 
    ws:    *mut Workspace, 
    level: Option<i32>) -> NetDef 
{
    let level = level.unwrap_or(1);
    
    todo!();
    /*
        auto nn = convertToNNModule(net);
      graphOptimzations(&nn, level);
      workspaceOptimizations(&nn, ws, level);
      return convertToCaffe2Proto(nn, net);
    */
}

#[inline] pub fn optimize(
    net:   NetDef, 
    level: Option<i32>) -> NetDef 
{
    let level = level.unwrap_or(1);
    
    todo!();
    /*
        auto nn = convertToNNModule(net);
      graphOptimzations(&nn, level);
      return convertToCaffe2Proto(nn, net);
    */
}
