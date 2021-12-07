
class BilstmCrfConfig():
    def __init__(self, config):
        self.embedding_size = config.embedding_size
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.dropout_prob = config.dropout_prob
        self.num_labels = config.num_labels
