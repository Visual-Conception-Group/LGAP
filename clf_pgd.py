import foolbox
import torch

### Classifer PGD attack
# Attack input x
def clf_pgd(x, y, network_clf, device='cuda', ptb=8., dim=-1):
    x = x.to(device)
    y = y.to(device)
    fmodel = foolbox.PyTorchModel(network_clf, bounds=(0., 1.))
    if dim==-1:
        attack = foolbox.attacks.LinfPGD(rel_stepsize=0.25, steps=40) # Can be modified for better attack
        _, x_adv, success = attack(fmodel, x, y, epsilons=ptb/256.)
        acc = 1 - success.float().mean(axis=-1)
    elif dim==2:
        attack = foolbox.attacks.L2PGD(rel_stepsize=0.25) # Can be modified for better attack
        _, x_adv, success = attack(fmodel, x, y, epsilons=ptb/256.)
        acc = 1 - success.float().mean(axis=-1)
    return x_adv, success, acc