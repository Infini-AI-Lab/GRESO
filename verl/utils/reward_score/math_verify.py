# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig


def compute_score(model_output: str, ground_truth: str) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception as e:
        print(e)

    return ret_score

if __name__ == '__main__':
    solution_str = "Assistant: Here is the solution to the problem. \\boxed{\\frac{1}{2} + 32c }"
    ground_truth = "\\frac{1}{2} + 5c + 27c"
    print(solution_str, ground_truth)
    print(compute_score(solution_str, ground_truth)) #1.0
    ground_truth = "\\frac{1}{3}"
    print(solution_str, ground_truth)
    print(compute_score(solution_str, ground_truth)) #0.1

    solution_str = "The answer is \\boxed{1/2}"
    ground_truth = "\\frac{1}{2}"
    print(solution_str, ground_truth)
    print(compute_score(solution_str, ground_truth)) #0