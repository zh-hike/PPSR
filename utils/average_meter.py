import paddle

class AverageMeter:
    def __init__(self, name, postfix="", round=4):
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
        if isinstance(val, paddle.Tensor):
            val = val.item()
        self.val = val * n
        self.count += n
        self.total += self.val
        self.avg = self.total / self.count

    @property
    def info(self):
        s = f"%.{self.round}f"%self.avg
        return f"{self.name}: {s}{self.postfix} "


class AverageMeterDict:
    def __init__(self, names, postfixs=None, round=4):
        if postfixs is None:
            postfixs = ["" for _ in range(len(names))]
        self.amd = {}
        for name, postfix in zip(names, postfixs):
            self.amd[name] = AverageMeter(name, postfix, round=round)

    def reset(self):
        for _, v in self.amd.items():
            v.reset()

    def update(self, vals, n=1):
        assert isinstance(vals, dict), f"vals 必须是dict格式，不能是{type(vals)}格式"
        for key, val in vals.items():
            if not key in self.amd:
                self.amd[key] = AverageMeter(key)
            self.amd[key].update(val)

    @property
    def info(self):
        s = ''
        for _, am in self.amd.items():
            s += am.info
        return s
