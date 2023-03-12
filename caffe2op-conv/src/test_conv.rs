crate::ix!();

#[test] fn conv_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Conv",
        ["X", "filter", "bias"],
        ["Y"],
        kernel=5,
        pad=1,
        stride=2
    )

    // Create X: (N,C,H,W)
    data = np.random.randn(1,1,8,8).astype(np.float32)
    print("Data shape: ",data.shape)

    // Create W: (M,C,Kh,Kw)
    filters = np.random.randn(3,1,5,5).astype(np.float32)
    print("Filter shape: ",filters.shape)

    // Create b: M
    bias = np.array([1.,1.,1.]).astype(np.float32)
    print("Bias shape: ",bias.shape)

    // Put the inputs into the workspace
    workspace.FeedBlob("X", data)
    workspace.FeedBlob("filter", filters)
    workspace.FeedBlob("bias", bias)

    // Run the operator
    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    Data shape:  (1, 1, 8, 8)
    Filter shape:  (3, 1, 5, 5)
    Bias shape:  (3,)
    Y:
     [[[[  0.6406407    0.8620521    0.56461596]
       [ -1.5042953   -0.79549205 -10.683343  ]
       [ -0.5240259    3.4538248   -3.9564204 ]]

      [[  0.6876496    4.8328524   -1.9525816 ]
       [  1.2995434   -2.3895378    7.2670045 ]
       [  3.9929862    1.8126237    5.4699917 ]]

      [[  3.55949      4.7934155    0.76086235]
       [  3.9588015   -1.3251319    4.413117  ]
       [ -1.5296054   -1.4924102   -3.2552304 ]]]]

    */
}
