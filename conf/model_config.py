
class BilstmCrfConfig():
    def __init__(self, config):
        embedding_size = config.embedding_size
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        dropout_prob = config.dropout_prob
        num_labels = config.num_labels
