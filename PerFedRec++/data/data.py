class Data(object):
    def __init__(self, conf, training, test, valid):
        self.config = conf
        self.training_data = training
        self.test_data = test #can also be validation set if the input is for validation
        self.valid_data = valid







