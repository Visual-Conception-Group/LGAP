import foolbox
import torch

### BPDA+EOT attack
# Attack input x
def bpda_strong(x, y, network_ebm, network_clf, device="cuda", attack_iter=40, n_eot=1, ptb=8., alpha=2., dim=-1):
    fmodel = foolbox.PyTorchModel(network_clf.to(torch.float16), bounds=(0., 1.))
    x = x.to(device)
    y = y.to(device)
    x_temp = x.clone().detach()
    for i in range(attack_iter):
        # get gradient of purified images for n_eot times
        grad = torch.zeros_like(x_temp).to(device)
        for j in range(n_eot):
            x_temp_eot = network_ebm(x_temp)
            if dim==-1:
                attack = foolbox.attacks.LinfPGD(rel_stepsize=0.25, steps=1, random_start=False) # Can be modified for better attack
                _, x_temp_eot_d, _ = attack(fmodel, x_temp_eot.to(torch.float16), y, epsilons=ptb/255.)
            elif dim==2:
                attack = foolbox.attacks.L2PGD(rel_stepsize=0.25) # Can be modified for better attack
                _, x_temp_eot_d, _ = attack(fmodel, x_temp_eot, y, epsilons=ptb/255.)
            grad += (x_temp_eot_d.detach() - x_temp_eot).to(device)
        # Check attack success
        x_clf = x_temp.clone().detach().to(device)
        success = torch.eq(torch.argmax(network_clf(x_clf), dim=1), y)
        #grad *= success[:, None, None, None] # Attack correctly classified images only
        x_temp = torch.clamp(x + torch.clamp(x_temp - x + grad.sign()*alpha/255., -1.0*ptb/255., ptb/255.), 0.0, 1.0)

    x_adv = x_temp.clone().detach()
    x_clf = x_adv.clone().detach().to(device)
    success = torch.eq(torch.argmax(network_clf(x_clf), dim=1), y)
    acc = success.float().mean(axis=-1)

    return x_adv, success, acc