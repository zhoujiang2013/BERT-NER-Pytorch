
class BilstmCrfConfig():
    def __init__(self, config):
        embedding_size = config.embedding_size
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        hidden_dropout_prob = config.hidden_dropout_prob
        num_labels = config.num_labels
