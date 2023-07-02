class EarlyStop(object) :
    
    def __init__(self,
                 patience = 10) :
        self.patience = patience
        self.is_stop = False
        self.counter = 0
        self.best_loss = 1e5
        self.is_best = False
        
    def __call__(self, loss) :
        if loss < self.best_loss :
            self.best_loss = loss
            self.counter = 0
            self.is_best = True
        else :
            self.counter += 1
            self.is_best = False
            
        if self.counter >= self.patience :
            self.is_stop = True
            
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)