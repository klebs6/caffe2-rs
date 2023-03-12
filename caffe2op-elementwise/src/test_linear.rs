crate::ix!();

#[test] fn elementwise_linear_op_example() {
    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ElementwiseLinear",
        ["X", "w", "b"],
        ["Y"]
    )

    // Create X
    X = np.array([[1,2,3,4,5],[6,8,9,16,10]])
    print("X:\n",X)

    // Create w
    w = np.array([1,1/2.,1/3.,1/4.,1/5.])
    print("w:\n",w)

    // Create b
    b = np.array([1.,1.,1.,1.,1.])
    print("b:\n",b)


    // Feed X & w & b into workspace
    workspace.FeedBlob("X", X.astype(np.float32))
    workspace.FeedBlob("w", w.astype(np.float32))
    workspace.FeedBlob("b", b.astype(np.float32))

    // Run op
    workspace.RunOperatorOnce(op)

    // Collect Output
    print("Y:\n", workspace.FetchBlob("Y"))

    **Result**

    X:
     [[ 1  2  3  4  5]
     [ 6  8  9 16 10]]
    w:
     [1.  0.5  0.33333333 0.25  0.2]
    b:
     [1. 1. 1. 1. 1.]
    Y:
     [[2. 2. 2. 2. 2.]
     [7. 5. 4. 5. 3.]]

    */
}
