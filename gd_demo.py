import sys
lr = float(sys.argv[1])
w = 0.0                      # start far from the best value (3.0)
for step in range(10):
    grad = 2 * (w - 3)       # slope of the error at the current w
    w = w - lr * grad        # step against the slope, scaled by lr
    print(f"step {step+1}: w = {w:.3f}")
