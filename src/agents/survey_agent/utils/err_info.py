


class CumulativeErrorInfo:
    def __init__(self, maxstore: int = 10):
        self.maxstore = maxstore
        self.errors = []

    def add_error(self, error_msg: str):
        if error_msg in self.errors:
            return
        self.errors.append(error_msg)
        if len(self.errors) > self.maxstore:
            self.errors.pop(0)

    def add_errors(self, error_msgs):
        for msg in error_msgs:
            self.add_error(msg)

    def get_errors(self):
        return self.errors.copy() 

    def get_errors_text(self):
        return "Errors in previous attempts:\n" + "\n".join(self.errors)