import pytest
import torch

from foundations.micrograd.engine import Value

# approx uses max(rel*|expected|, abs), so abs must be as tight as rel; float64 gives ~1e-16.
TOLERANCE_REL = 1e-12
TOLERANCE_ABS = 1e-12


def test_gradient_vs_pytorch():
    # Arrange
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b = Value(6.8813735870195432)

    p_x1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    p_x2 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    p_w1 = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    p_w2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    p_b = torch.tensor([6.8813735870195432], dtype=torch.float64, requires_grad=True)

    # Test
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.backward()

    p_n = p_x1 * p_w1 + p_x2 * p_w2 + p_b
    p_o = torch.tanh(p_n)
    p_o.backward()

    # Assert
    assert o.data == pytest.approx(p_o.data, rel=TOLERANCE_REL, abs=TOLERANCE_ABS)

    assert p_x1.grad is not None
    assert x1.grad == pytest.approx(
        p_x1.grad.item(), rel=TOLERANCE_REL, abs=TOLERANCE_ABS
    )

    assert p_x2.grad is not None
    assert x2.grad == pytest.approx(
        p_x2.grad.item(), rel=TOLERANCE_REL, abs=TOLERANCE_ABS
    )

    assert p_w1.grad is not None
    assert w1.grad == pytest.approx(
        p_w1.grad.item(), rel=TOLERANCE_REL, abs=TOLERANCE_ABS
    )

    assert p_w2.grad is not None
    assert w2.grad == pytest.approx(
        p_w2.grad.item(), rel=TOLERANCE_REL, abs=TOLERANCE_ABS
    )

    assert p_b.grad is not None
    assert b.grad == pytest.approx(
        p_b.grad.item(), rel=TOLERANCE_REL, abs=TOLERANCE_ABS
    )


def test_micrograd_div_vs_pytorch():
    # Arrange
    a = Value(1 / 3)
    p_a = torch.tensor([1 / 3], dtype=torch.float64, requires_grad=True)

    # Test
    o = 1 / a
    p_o = 1 / p_a

    o.backward()
    p_o.backward()

    # Assert
    assert o.data == pytest.approx(
        p_o.data.item(), rel=TOLERANCE_REL, abs=TOLERANCE_ABS
    )

    assert a.grad == pytest.approx(
        p_a.grad.item(), rel=TOLERANCE_REL, abs=TOLERANCE_ABS
    )


def test_gradient_accumulation_repeated_variable():
    # Arrange
    a = Value(3)
    b = a * a
    c = Value(10)
    s = b + a

    # Test
    n = s * c
    o = 5 * n

    n.backward()

    # Assert
    assert n.grad == 1
    assert c.grad == 12
    assert s.grad == 10
    assert b.grad == 10
    assert a.grad == 70

    o.backward()

    # Assert
    assert o.grad == 1
    assert n.grad == 6
    assert c.grad == 84
    assert s.grad == 70
    assert b.grad == 80
    assert a.grad == 620

    n.zero_grad()
    c.zero_grad()
    s.zero_grad()
    b.zero_grad()
    a.zero_grad()

    o.backward()

    # Assert
    assert o.grad == 1
    assert n.grad == 5
    assert c.grad == 60
    assert b.grad == 50
    assert a.grad == 350
