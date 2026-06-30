from __future__ import annotations

import pytest

from dr_dspy.humaneval.code_extraction import (
    apply_cleaning,
    validate_python_source,
)
from dr_dspy.humaneval.import_inference import infer_necessary_imports


@pytest.mark.parametrize(
    ("source", "expected_prefix"),
    [
        (
            "def f(x):\n    return np.array(x)\n",
            "import numpy as np\n",
        ),
        (
            "def f():\n    return math.sqrt(2)\n",
            "import math\n",
        ),
        (
            "def f():\n    return Counter([1, 1])\n",
            "from collections import Counter\n",
        ),
        (
            "def f():\n    return nn.Linear(1, 1)\n",
            "import torch.nn as nn\n",
        ),
    ],
)
def test_infer_necessary_imports_prepends_missing_imports(
    source: str,
    expected_prefix: str,
) -> None:
    result = infer_necessary_imports(source)
    assert result.startswith(expected_prefix)
    assert source.strip() in result


def test_infer_necessary_imports_skips_existing_import() -> None:
    source = "import numpy as np\n\ndef f(x):\n    return np.array(x)\n"
    result = infer_necessary_imports(source)
    assert result.count("import numpy as np") == 1
    assert "def f(x):" in result


def test_infer_necessary_imports_skips_locally_bound_name() -> None:
    source = "np = 1\n\ndef f():\n    return np\n"
    result = infer_necessary_imports(source)
    assert "import numpy" not in result
    assert "return np" in result


def test_infer_necessary_imports_adds_multiple_missing_imports() -> None:
    source = "def f():\n    return np.zeros(1) + math.pi\n"
    result = infer_necessary_imports(source)
    assert "import numpy as np" in result
    assert "import math" in result
    assert result.index("import numpy as np") < result.index("import math")


def test_infer_necessary_imports_repairs_trailing_comment_on_import_line() -> (
    None
):
    source = (
        "from collections import (Counter,  # noqa\n"
        "\n"
        "def f():\n"
        "    return Counter([1])\n"
    )
    result = infer_necessary_imports(source)
    assert "from collections import (Counter)" in result
    assert "def f():" in result


def test_infer_necessary_imports_repairs_unbalanced_import_parens() -> None:
    source = (
        "from typing import (List, Dict\n"
        "\n"
        "def f():\n"
        "    return List[int]\n"
    )
    result = infer_necessary_imports(source)
    assert "from typing import (List, Dict)" in result


def test_infer_necessary_imports_deduplicates_import_lines() -> None:
    source = "import math\nimport math\n\ndef f():\n    return math.pi\n"
    result = infer_necessary_imports(source)
    assert result.count("import math") == 1


def test_infer_necessary_imports_passthrough_on_syntax_error() -> None:
    source = "def f(x\n    return np.array(x)\n"
    result = infer_necessary_imports(source)
    assert "return np.array(x)" in result
    assert "import numpy" not in result


def test_infer_necessary_imports_ignores_unmapped_names() -> None:
    source = "def f():\n    return random.randint(0, 1)\n"
    result = infer_necessary_imports(source)
    assert "random.randint" in result
    assert "import random" not in result


def test_apply_cleaning_infers_imports_for_compilable_candidate() -> None:
    source = "```python\ndef f(x):\n    return np.array([x])\n```"
    candidates = apply_cleaning(source, apply_dedent=True)
    assert candidates
    assert candidates[0].startswith("import numpy as np")
    assert validate_python_source(candidates[0]).compile_ok
