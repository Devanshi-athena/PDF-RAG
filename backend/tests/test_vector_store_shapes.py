import math

from backend.vector_store import HFEmbeddingFunction


class FakeResult:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def test_single_item_batch_response_is_accepted():
    result = FakeResult([[1.0] * 384])

    vectors = HFEmbeddingFunction._convert_output(result, 1)

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert math.isclose(math.sqrt(sum(value * value for value in vectors[0])), 1.0)
