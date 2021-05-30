class Config:
    ## Frames
    skip_frames = 3
    history_length = 0

    ## Optimzation
    learning_rate = 5e-5
    batch_size = 16
    n_epochs = 10
    validation_frac = 0.3

    ## testing
    n_test_episodes = 15
    rendering = False

    ## (FCN or CNN)
    is_fcn = False
    
    dev_size = 1000
