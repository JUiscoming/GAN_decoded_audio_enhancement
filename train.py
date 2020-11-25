from torch.utils.data import DataLoader

def train_GAN(G, D, dataset, **opt):
    # Dataloader
    dataloader = DataLoader(dataset = dataset, batch_size = opt['batch_size'], shuffle = True)
    iterations = dataloader.dataset.__len__() // opt['batch_size']
    
    # 
    noise_shape = (opt['batch_size'], ) if 'reduced_model'
    
    for epoch in range(loaded_epoch, epochs):
        
        for iteration, (original, decoded) in enumerate(dataloader):
            if iteration == iterations:
                break
            G.train()
            original, decoded = original.to(opt['device']), decoded.to(opt['device'])
            
            if opt['Enhancement'] == True: # True: Learn a mapping decoded to original. False: Learn a mapping original to decoded.
                mapped = G(decoded, z)
            elif opt['Enhancement'] == False:
                mapped = G(original, z)
            