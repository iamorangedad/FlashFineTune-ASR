from autograd.node import Value
from autograd.sgd import SGD


def test_sgd():
    xs = [Value(1.0), Value(3.0)]
    ys = [Value(3.0), Value(7.0)]
    a = Value(1.0)
    b = Value(1.0)
    optimizer = SGD(params=[a, b], lr=0.05)
    for epoch in range(5):
        total_loss = Value(0.0)
        for i in range(len(xs)):
            predict = a * xs[i] + b
            diff = predict + Value(-1) * ys[i]
            loss = diff * diff
            total_loss += loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(
            f"Epoch {epoch+1}: Loss={total_loss.data:.6f} | a.grad={a.grad:.2f}, b.grad={b.grad:.2f} | result a={a.data:.4f}, b={b.data:.4f}"
        )
    print(f"Final model: y = {a.data:.4f} * x + {b.data:.4f}")
