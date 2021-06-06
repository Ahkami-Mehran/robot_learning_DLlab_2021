class Config:
    ## Frames
    skip_frames = 2 # 4 next 
    history_length = 0 # 1 3 5 1 3 5

    ## Optimzation
    learning_rate = 5e-5
    batch_size = 32
    n_epochs = 10
    validation_frac = 0.3

    ## testing
    n_test_episodes = 15
    rendering = False

    ## (FCN or CNN)
    is_fcn = False
    
    dev_size = 1000
