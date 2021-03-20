class DepParseAugmentor:
    def __init__(self, src_input_path, tgt_input_path, augment=True):
        self.src_input_path = src_input_path
        self.tgt_input_path = tgt_input_path

        self.src_input_file = open(self.src_input_path, 'r')
        self.tgt_input_file = open(self.tgt_input_path, 'r')

        self.src_sent = None
        self.tgt_sent = None

    def __iter__(self):
        self.src_sent = next(self.src_input_line)
        self.tgt_sent = next(self.tgt_input_line)

        return self.src_sent, self.tgt_sent

    def __next__(self):
        pass