crate::ix!();

#[test] fn cross_entropy_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "CrossEntropy",
        ["X", "label"],
        ["Y"]
    )

    // Create X: Sample softmax output for 5-class model
    X = np.array([[.01, .05, .02, .02, .9],[.03, .1, .42, .05, .4]])
    print("X:\n",X)

    // Create label: Sample 1-hot ground truth label vectors
    label = np.array([[0.,0.,0.,0.,1.],[0.,0.,1.,0.,0.]])
    print("label:\n",label)

    // Feed X & label into workspace
    workspace.FeedBlob("X", X.astype(np.float32))
    workspace.FeedBlob("label", label.astype(np.float32))

    // Run op
    workspace.RunOperatorOnce(op)

    // Collect Output
    print("Y:\n", workspace.FetchBlob("Y"))


    X:
     [[0.01 0.05 0.02 0.02 0.9 ]
     [0.03 0.1  0.42 0.05 0.4 ]]
    label:
     [[0. 0. 0. 0. 1.]
     [0. 0. 1. 0. 0.]]
    Y:
     [0.10536055 0.8675006 ]

    */
}
