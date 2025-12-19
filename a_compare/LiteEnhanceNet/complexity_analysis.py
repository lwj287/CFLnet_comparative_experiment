import torch
from thop import profile
from torchsummary import summary
import time
import numpy as np
from model import Mynet

if __name__ == "__main__":
    model = Mynet().cuda()
    model.eval()
    input = torch.randn(1, 3, 1920, 1080).cuda()

    # FPS
    process_time = []
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = model(input)
            torch.cuda.synchronize()
            end_time = time.time()
            process_time.append(end_time - start_time)
    print(f"FPS: {(1. / np.mean(process_time[1:])):.3f} f/s")

    # FLOPs
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")

    # Memory
    summary(model, (3, 1920, 1080))