import torch
import tqdm as tqdm

def train(loader, model, crit, optimizer, sched=None, device='cpu'):
    model.train()
    t = tqdm.tqdm(loader, leave=False, total=len(loader))
    losses = []
    ops = []
    targs = []

    for i, input in enumerate(t):
        ids = torch.as_tensor(input['input_ids'], dtype=torch.long).clone().detach()
        masks = torch.as_tensor(input['attention_mask'], dtype=torch.long).clone().detach()
        token_type_ids = torch.as_tensor(input['token_type_ids'], dtype=torch.long).clone().detach()
        targets = torch.as_tensor(input['targets'], dtype=torch.long).clone().detach()

        ids, masks = ids.to(device), masks.to(device)
        token_type_ids, targets = token_type_ids.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(ids, masks, token_type_ids)

        loss = crit(outputs, targets)
        losses.append(loss.item())

        loss.backward()

        optimizer.step()
        if sched:
            sched.step()

        op = torch.max(outputs.data, 1)[1]
        ops.extend(op.cpu().detach().numpy().tolist())
        targs.extend(targets.cpu().detach().numpy().tolist())

    return losses, ops, targs



def test(loader, model, crit, device):
    model.eval()
    t = tqdm.tqdm(loader, leave=False, total=len(loader))
    losses = []
    ops = []
    targs = []

    with torch.no_grad():
        for i, input in enumerate(t):
            ids = torch.as_tensor(input['input_ids'], dtype=torch.long).clone().detach()
            masks = torch.as_tensor(input['attention_mask'], dtype=torch.long).clone().detach()
            token_type_ids = torch.as_tensor(input['token_type_ids'], dtype=torch.long).clone().detach()
            if 'targets' in input.keys() and input['targets'] is not None:
                targets = torch.as_tensor(input['targets'], dtype=torch.long).clone().detach()
                targets = targets.to(device)

            ids, masks, token_type_ids = ids.to(device), masks.to(device), token_type_ids.to(device)

            outputs = model(ids, masks, token_type_ids)
            op = torch.max(outputs.data, 1)[1]
            
            if 'targets' in input.keys() and input['targets'] is not None:
                loss = crit(outputs, targets)
                losses.append(loss)
                targs.extend(targets.cpu().detach().numpy().tolist())
                
            ops.extend(op.cpu().detach().numpy().tolist())

    return losses, ops, targs
            


