from typing import Any

from flow_judge.metrics.metric import RubricItem

USER_PROMPT_TEMPLATE = """# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large \
language model.

You will be provided with the inputs and output of the task, as well as the evaluation criteria \
and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation \
criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{INPUTS}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{OUTPUT}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
<evaluation_criteria>
{EVALUATION_CRITERIA}
</evaluation_criteria>

<scoring_rubric>
{RUBRIC}
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. \
Review the evaluation criteria and scoring rubric to understand the different levels of \
performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output \
generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score \
descriptions in the scoring rubric. For each criterion,decide which description best matches the \
output.
4. After comparing the output to the score descriptions, pay attention to the small details that \
might impact the final score that you assign. Sometimes a small difference can dictate the final \
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring \
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always \
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."""


USER_PROMPT_NO_INPUTS_TEMPLATE = """# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large language model.

You will be provided the output of the task, as well as the evaluation criteria \
and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation \
criteria and scoring rubric provided.

# OUTPUT
Below is the output of the task:
<output>
{OUTPUT}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
<evaluation_criteria>
{EVALUATION_CRITERIA}
</evaluation_criteria>

<scoring_rubric>
{RUBRIC}
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. \
Review the evaluation criteria and scoring rubric to understand the different levels of \
performance and the descriptions for each score.
2. Review the output: Examine the output generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score \
descriptions in the scoring rubric. For each criterion,decide which description best matches the \
output.
4. After comparing the output to the score descriptions, pay attention to the small details that \
might impact the final score that you assign. Sometimes a small difference can dictate the final \
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring \
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always \
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."""


def format_vars(variables: list[dict[str, str]]) -> str:
    """Format variables for the prompt."""
    var_strs = []
    for var in variables:
        for key, value in var.items():
            var_tag = key.lower().replace(" ", "_")
            var_strs.append(f"<{var_tag}>\n{value}\n</{var_tag}>")
    return "\n".join(var_strs)


def format_rubric(rubric: list[RubricItem]) -> str:
    """Format the rubric for the prompt."""
    rubric_strs = []

    # Sort rubric items by score, lowest to highest
    sorted_rubric = sorted(rubric, key=lambda x: x.score)

    for item in sorted_rubric:
        rubric_strs.append(f"- Score {item.score}: {item.description}")
    return "\n".join(rubric_strs)


def format_user_prompt(prompt_variables: dict[str, Any]) -> str:
    """Format the user prompt based on provided variables."""
    if prompt_variables["INPUTS"]:
        return USER_PROMPT_TEMPLATE.format(**prompt_variables)
    else:
        return USER_PROMPT_NO_INPUTS_TEMPLATE.format(**prompt_variables)
