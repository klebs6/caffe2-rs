crate::ix!();

#[test] fn instance_norm_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "InstanceNorm",
        ["input", "scale", "bias"],
        ["output"],
        epsilon=1e-5,
    )

    workspace.FeedBlob("input", np.random.randn(2, 1, 3, 3).astype(np.float32))
    print("input:\n", workspace.FetchBlob("input"), "\n")

    workspace.FeedBlob("scale", np.array([1.5]).astype(np.float32))
    print("scale: ", workspace.FetchBlob("scale"))

    workspace.FeedBlob("bias", np.array([1.]).astype(np.float32))
    print("bias: ", workspace.FetchBlob("bias"))

    workspace.RunOperatorOnce(op)
    print("output:\n", workspace.FetchBlob("output"))

    input:
     [[[[ 0.97856593 -1.1832817  -0.2540021 ]
       [-1.3315694  -0.7485018   0.3787225 ]
       [-0.6826597  -1.4637762   0.57116514]]]


     [[[-0.44948956  0.85544354 -0.9315333 ]
       [-0.37202677 -0.22266895 -0.27194235]
       [ 0.4948163  -0.7296504   1.3393803 ]]]]

    scale:  [1.5]
    bias:  [1.]
    output:
     [[[[ 3.5017493  -0.3791256   1.2890853 ]
       [-0.6453266   0.40137637  2.4249308 ]
       [ 0.5195738  -0.8826599   2.7703972 ]]]


     [[[ 0.12639964  2.856744   -0.8821926 ]
       [ 0.28847694  0.60098207  0.49788612]
       [ 2.1021945  -0.45978796  3.869297  ]]]]
    */
}
