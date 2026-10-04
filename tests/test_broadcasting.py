import pytest
import numpy as np
from hypothesis import given, strategies as st
from tinytorch.tensor import Tensor, unbroadcast


# Helper strategy to generate broadcastable shape pairs
@st.composite
def broadcastable_shapes(draw):
    ndim = draw(st.integers(min_value=1, max_value=4))
    shape1 = []
    shape2 = []
    for _ in range(ndim):
        dim1 = draw(st.integers(min_value=1, max_value=5))
        # Choose to match dim1, be 1, or be equal
        choice = draw(st.sampled_from(["same", "one1", "one2"]))
        if choice == "same":
            shape1.append(dim1)
            shape2.append(dim1)
        elif choice == "one1":
            shape1.append(1)
            shape2.append(dim1)
        else:
            shape1.append(dim1)
            shape2.append(1)
    # Optionally drop leading dimensions from one shape
    if len(shape1) > 1 and draw(st.booleans()):
        drop_count = draw(st.integers(min_value=1, max_value=len(shape1) - 1))
        shape2 = shape2[drop_count:]

    return tuple(shape1), tuple(shape2)


@given(shapes=broadcastable_shapes())
def test_hypothesis_broadcasting_add(shapes):
    shape1, shape2 = shapes
    arr1 = np.random.randn(*shape1).astype(np.float32)
    arr2 = np.random.randn(*shape2).astype(np.float32)

    t1 = Tensor(arr1, requires_grad=True)
    t2 = Tensor(arr2, requires_grad=True)

    out = (t1 + t2).sum()
    out.backward()

    # Forward check
    expected_out = arr1 + arr2
    assert t1.grad is not None
    assert t2.grad is not None
    assert t1.grad.shape == t1.shape
    assert t2.grad.shape == t2.shape


@given(shapes=broadcastable_shapes())
def test_hypothesis_broadcasting_mul(shapes):
    shape1, shape2 = shapes
    arr1 = np.random.randn(*shape1).astype(np.float32)
    arr2 = np.random.randn(*shape2).astype(np.float32)

    t1 = Tensor(arr1, requires_grad=True)
    t2 = Tensor(arr2, requires_grad=True)

    out = (t1 * t2).sum()
    out.backward()

    assert t1.grad.shape == t1.shape
    assert t2.grad.shape == t2.shape


def test_unbroadcast_explicit():
    grad = np.ones((3, 4), dtype=np.float32)
    reduced_x = unbroadcast(grad, (4,))
    assert reduced_x.shape == (4,)
    np.testing.assert_allclose(reduced_x, [3.0, 3.0, 3.0, 3.0])

    reduced_y = unbroadcast(grad, (3, 1))
    assert reduced_y.shape == (3, 1)
    np.testing.assert_allclose(reduced_y, [[4.0], [4.0], [4.0]])
