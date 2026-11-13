import time, torch, torch.nn as nn

def make_net():
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512), nn.ReLU(),   # 64 * 7 * 7 = 3136 after the convs
        nn.Linear(512, 7))

def sync(device):
    if device == "mps":
        torch.mps.synchronize()            # wait for the async GPU to finish

def bench(device, batch, train, iters=50):
    net = make_net().to(device)
    opt = torch.optim.Adam(net.parameters(), 2.5e-4)
    x = torch.rand(batch, 4, 84, 84, device=device)
    def step():
        if train:
            loss = net(x).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            with torch.no_grad():
                net(x)
    for _ in range(5):
        step()                              # warm-up, not timed
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000

print(f"{'what':<22}{'cpu ms':>10}{'mps ms':>10}")
for label, batch, train in [("inference batch 1", 1, False),
                            ("inference batch 4", 4, False),
                            ("train batch 64", 64, True),
                            ("train batch 256", 256, True)]:
    print(f"{label:<22}{bench('cpu', batch, train):>10.2f}{bench('mps', batch, train):>10.2f}")
