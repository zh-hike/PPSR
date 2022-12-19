
class AverageMeter:
    def __init__(self, name, postfix, round=4):
        self.name = name
        self.postfix = postfix
        self.reset()
        self.round = round

    def reset(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val * n
        self.count += n
        self.total += self.val
        self.avg = self.total / self.count

    @property
    def info(self):
        return f"{name}_Avg: {round(self.avg, self.round)}{postfix} "