import foolbox
import torch
# from utils import *
# from purification import *

### Classifer PGD attack
# Attack input x
def bpda(x, y, network_ebm, network_clf, device="cuda", ball_dim=-1, ptb=8., alpha=2., n_eot=1, iterations=40):
    fmodel = foolbox.PyTorchModel(network_clf, bounds=(0., 1.))
    x = x.to(device).to(torch.float16)
    y = y.to(device)
    x_temp = x.clone().detach()
    for i in range(iterations):
        # get gradient of purified images for n_eot times
        grad = torch.zeros_like(x_temp).to(device)
        for j in range(n_eot):
            x_temp_eot = network_ebm(x_temp)
            if ball_dim==-1:
                attack = foolbox.attacks.LinfPGD(rel_stepsize=0.05, steps=1, random_start=False) # Can be modified for better attack
                _, x_temp_eot_d, _ = attack(fmodel, x_temp_eot, y, epsilons=ptb/255.)
            elif ball_dim==2:
                attack = foolbox.attacks.L2PGD(rel_stepsize=0.05) # Can be modified for better attack
                _, x_temp_eot_d, _ = attack(fmodel, x_temp_eot.float(), y, epsilons=ptb/255.)
            grad += (x_temp_eot_d.detach() - x_temp_eot).to(device)
        # Check attack success
        x_clf = x_temp.clone().detach().to(device)
        success = torch.eq(torch.argmax(network_clf(x_clf), dim=1), y)
        grad *= success[:, None, None, None] # Attack correctly classified images only
        x_temp = torch.clamp(x + torch.clamp(x_temp - x + grad.sign()*alpha/255., -1.0*ptb/255., ptb/255.), 0.0, 1.0)

    x_adv = x_temp.clone().detach()
    x_clf = x_adv.clone().detach().to(device)
    success = torch.eq(torch.argmax(network_clf(network_ebm(x_clf)), dim=1), y)
    acc = success.float().mean(axis=-1)

    return x_adv, success, acc