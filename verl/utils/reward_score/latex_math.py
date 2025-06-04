import re
from typing import Dict
import random

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
# from verl.utils.reward_score.math_dataset import remove_boxed, last_boxed_only_string

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def extract_solution(solution_str):
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            return remove_boxed(string_in_last_boxed)
        else:
            return None
    except Exception as e:
        print(e)
        print("@@@"*20)
        print(solution_str)
        print("+++"*20)
        print("Failed to extract answer from solution string.")
        print("@@@"*20)
        return None

def find_boxed_content(text):
    """
    Detect if a string contains '\boxed{...}' and extract the content inside.

    Args:
        text (str): The input string.

    Returns:
        tuple: (True, content) if a boxed expression is found, else (False, None).
    """
    pattern = r'\\boxed\{(.*?)\}'
    match = re.search(pattern, text)

    if match:
        return True, match.group(1)  # Extract content inside \boxed{}
    return False, None


def compute_score(prompt_str, solution_str, ground_truth, format_score=0.1, full_score=1.) -> float:
    try:
        do_print = random.randint(1, 4096) == 1
        if solution_str is None: return 0.0

        ans_string = extract_solution(solution_str)
        if do_print:
            print("\n" + "="*80)
            print(" Processing New Sample ".center(80, '='))
            print(f">> Prompt: {prompt_str}")
            print("-"*80)
            print(f">> Output: {solution_str}")
            print(f">> Extracted Answer: {ans_string}")
            print(f">> Ground Truth: {ground_truth}")

        if ans_string is None:
            if do_print: print(f">> Failed to extract answer from solution string.")
            if do_print: print(f">> Score: {0.0}")
            return 0.0

        if math_is_equivalent(ans_string, ground_truth):
            if do_print: print(f">> Answer validation: FULL MATCH")
            if do_print: print(f">> Score: {full_score}")
            return full_score
        else:
            if do_print: print(f">> Answer validation: MISMATCH")
            if do_print: print(f">> Score: {format_score}")
            # print("&"*80)
            # print("Ans: ", ans_string)
            # print("GT: ", ground_truth)
            # print("&"*80)
            return format_score
    except Exception as e:
        print(e)
        print("@@@"*20)
        print(solution_str)
        print("+++"*20)
        print("Failed to compute score.")
        print("@@@"*20)
        return 0.0

def wrap_math_in_latex(math_str):
    if type(math_str) is not str:
        math_str = str(math_str)
    math_str.strip()
    if not math_str.startswith("$"):
        math_str = f"${math_str}"
    if not math_str.endswith("$"):
        math_str = f"{math_str}$"
    return math_str

def math_is_equivalent(gold_answer, pred_answer):
    gold_answer = wrap_math_in_latex(gold_answer)
    pred_answer = wrap_math_in_latex(pred_answer)
    gold_answer = parse(gold_answer)
    pred_answer = parse(pred_answer)
    return verify(gold_answer, pred_answer)

def latex_equal(completion, solution)-> bool:
    """Reward function that checks if the completion is the same as the ground truth."""
    content = wrap_math_in_latex(completion)
    sol = wrap_math_in_latex(solution)

    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # print("Answer parsed: ", answer_parsed)
        reward = float(verify(answer_parsed, gold_parsed))
        return reward == 1.0
    else:
        # print("Failed to parse gold solution: ", sol)
        return False

#test the compute_score function
if __name__ == '__main__':
    solution_str = "Assistant: Here is the solution to the problem.\n <answer>\\frac{1}{2} + 32c </answer>"
    ground_truth = "\\frac{1}{2} + 5c + 27c"
    print(compute_score(solution_str, ground_truth)) #1.0
    ground_truth = "\\frac{1}{3}"
    print(compute_score(solution_str, ground_truth)) #0.1

    solution_str = "The answer is \\boxed{1/2}"
    ground_truth = "\\frac{1}{2}"
    print(compute_score(solution_str, ground_truth)) #0