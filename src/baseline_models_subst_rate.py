class BaselineSubstRate():
    def __init__(self):
        pass
        # super().__init__()
        
    def forward(self, x_musca, x_squirrel):
        subst = [x_musca[k]!=x_squirrel[k] for k in range(len(x_musca))]
        subst_rate = sum(subst) / len(subst)
        return subst_rate
