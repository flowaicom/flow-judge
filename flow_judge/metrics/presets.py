from .metric import Metric, RubricItem

# Pre-defined metrics
RESPONSE_CORRECTNESS_BINARY = Metric(
    name="Response Correctness (Binary)",
    criteria="""Does the generated response accurately match the provided reference answer \
for the given query?""",
    rubric=[
        RubricItem(
            score=0,
            description="""\
The generated response does not match the reference answer. It either contains inaccurate \
information, is missing key details from the reference, includes extra information not in the \
reference, or fails to convey the same meaning as the reference answer.""",
        ),
        RubricItem(
            score=1,
            description="""\
The generated response matches the reference answer exactly or contains all the key information \
from the reference with no inaccuracies, extra details, or missing details. The meaning conveyed \
by the generated response is equivalent to the reference.""",
        ),
    ],
    required_inputs=["query", "reference_answer"],
    required_output="response",
)

RESPONSE_CORRECTNESS_3POINT = Metric(
    name="Response Correctness (3-point Likert)",
    criteria="""\
Based on the provided reference response, how well does the system's generated response match the \
correct answer to the given query?""",
    rubric=[
        RubricItem(
            score=1,
            description="""\
The generated response does not match the reference response at all. It either fails to address \
the query or provides a completely incorrect answer.""",
        ),
        RubricItem(
            score=2,
            description="""\
The generated response partially matches the reference response. It addresses the query but may \
contain some incorrect, irrelevant or incomplete information compared to the reference.""",
        ),
        RubricItem(
            score=3,
            description="""\
The generated response fully matches the reference response. It accurately and completely answers \
the query, containing all the relevant information from the reference without any incorrect or \
extraneous details.""",
        ),
    ],
    required_inputs=["query", "reference_answer"],
    required_output="response",
)

RESPONSE_CORRECTNESS_5POINT = Metric(
    name="Response Correctness (5-point Likert)",
    criteria="""\
Compare the system's response to the provided reference answer and rate how well they match in \
accuracy and completeness to answer the query.""",
    rubric=[
        RubricItem(
            score=1,
            description="""\
The response is completely incorrect or irrelevant to the query, with no overlap in information \
with the reference answer.""",
        ),
        RubricItem(
            score=2,
            description="""\
The response contains some correct information relevant to the query but is substantially \
incomplete or inaccurate compared to the reference answer.""",
        ),
        RubricItem(
            score=3,
            description="""\
The response answers the query with reasonable accuracy but is missing key details or has minor \
inaccuracies compared to the reference.""",
        ),
        RubricItem(
            score=4,
            description="""\
The response accurately answers the query and is nearly complete, only leaving out non-essential \
details compared to the reference.""",
        ),
        RubricItem(
            score=5,
            description="""\
The response perfectly matches the accuracy and level of detail of the reference answer, \
containing all key information to comprehensively answer the query.""",
        ),
    ],
    required_inputs=["query", "reference_answer"],
    required_output="response",
)

RESPONSE_FAITHFULNESS_BINARY = Metric(
    name="Response Faithfulness (Binary)",
    criteria="""\
Based on the provided context, does the response contain only information that is supported by or \
directly inferable from the context?""",
    rubric=[
        RubricItem(
            score=0,
            description="""\
The response contains statements or claims that cannot be directly found in or logically inferred \
from the provided context. There is hallucinated or fabricated information present in the response \
that does not have support in the given context.""",
        ),
        RubricItem(
            score=1,
            description="""\
The response contains only statements and claims that are directly stated in or logically \
inferable from the provided context. There is no hallucinated or fabricated information present in \
the response that cannot be traced back to or deduced from the context.""",
        ),
    ],
    required_inputs=["query", "context"],
    required_output="response",
)

RESPONSE_FAITHFULNESS_3POINT = Metric(
    name="Response Faithfulness (3-point Likert)",
    criteria="""\
Based on the provided context, assess how faithful and consistent the response is to the \
information given. Check if the response contains any fabricated or hallucinated content that \
cannot be supported by the context.""",
    rubric=[
        RubricItem(
            score=1,
            description="""\
The response contains significant amount of fabricated information or unsupported claims that \
directly contradict or deviate from the given context. Major hallucinations are present that are \
not factual based on the context provided.""",
        ),
        RubricItem(
            score=2,
            description="""\
The response is mostly faithful to the context, but contains some minor unsupported details or \
slight factual inconsistencies. While the overall message is supported, there are a few deviations \
that are not directly inferable from the strict context alone.""",
        ),
        RubricItem(
            score=3,
            description="""\
The response is completely faithful and consistent with the context provided. All details and \
claims are directly supported by the information given, without any hallucinated or fabricated \
content present. The response accurately represents only the facts in the context.""",
        ),
    ],
    required_inputs=["query", "context"],
    required_output="response",
)

RESPONSE_FAITHFULNESS_5POINT = Metric(
    name="Response Faithfulness (5-point Likert)",
    criteria="""\
Based on the given context, evaluate how consistent and faithful the generated response is to the \
context. The response should not contain any hallucinated or fabricated information that is not \
supported by the context.""",
    rubric=[
        RubricItem(
            score=1,
            description="""\
The response is completely inconsistent with the provided context. It contains significant amount \
of hallucinated or fabricated information that directly contradicts or is not supported at all by \
the context.""",
        ),
        RubricItem(
            score=2,
            description="""\
The response is mostly inconsistent with the provided context. While it may contain some \
information from the context, it introduces a substantial amount of hallucinated or fabricated \
details that deviate from the context.""",
        ),
        RubricItem(
            score=3,
            description="""\
The response is somewhat consistent with the provided context. It includes a mix of information \
from the context and some hallucinated or fabricated details. The fabrications are minor and do \
not significantly contradict the context.""",
        ),
        RubricItem(
            score=4,
            description="""\
The response is mostly consistent with the provided context. The vast majority of the content is \
supported by the context, with only minor and inconsequential inconsistencies or fabrications, if \
any.""",
        ),
        RubricItem(
            score=5,
            description="""\
The response is completely consistent with and faithful to the provided context. All details in \
the response are directly supported by the context, without any hallucinated or fabricated \
information.""",
        ),
    ],
    required_inputs=["query", "context"],
    required_output="response",
)

RESPONSE_RELEVANCE_BINARY = Metric(
    name="Response Relevance (Binary)",
    criteria="""\
Is the response directly relevant to answering the query considering the context, without \
including irrelevant or extraneous information?""",
    rubric=[
        RubricItem(
            score=0,
            description="""\
The response does not sufficiently address the query, either by failing to directly answer the \
question asked, going off-topic, or including irrelevant or extraneous information that was not \
requested in the original query.""",
        ),
        RubricItem(
            score=1,
            description="""\
The response directly and sufficiently addresses the query. All of the content is relevant to \
answering the question asked, without going off-topic or providing unnecessary additional \
information beyond what the query requires.""",
        ),
    ],
    required_inputs=["query", "context"],
    required_output="response",
)

RESPONSE_RELEVANCE_3POINT = Metric(
    name="Response Relevance (3-point Likert)",
    criteria="""\
How relevant and pertinent is the response to addressing the given query, without including \
extraneous or irrelevant information?""",
    rubric=[
        RubricItem(
            score=1,
            description="""\
The response is not relevant to the query at all. It either does not address the key points of the \
query or includes only irrelevant or extraneous information that does not pertain to answering the \
query directly.""",
        ),
        RubricItem(
            score=2,
            description="""\
The response addresses some aspects of the query but is only partially relevant. It may go \
off-topic or include some tangentially related or extraneous information. Key points needed to \
comprehensively address the query are missing.""",
        ),
        RubricItem(
            score=3,
            description="""\
The response is highly relevant to the query and directly addresses all the key points needed to \
comprehensively answer the query. No irrelevant or extraneous information is included. The \
response is fully pertinent to the query.""",
        ),
    ],
    required_inputs=["query", "context"],
    required_output="response",
)

RESPONSE_RELEVANCE_5POINT = Metric(
    name="Response Relevance (5-point Likert)",
    criteria="""\
How well does the response address the query, providing relevant information without including \
anything extraneous or irrelevant?""",
    rubric=[
        RubricItem(
            score=1,
            description="""\
The response is completely irrelevant to the query, does not address it at all, or contains only \
extraneous information unrelated to the query.""",
        ),
        RubricItem(
            score=2,
            description="""\
The response is mostly irrelevant to the query, addressing it only tangentially or containing \
significant amounts of unrelated or extraneous information.""",
        ),
        RubricItem(
            score=3,
            description="""\
The response is somewhat relevant to the query, addressing the main point but going off-topic or \
including some extraneous details. Key aspects of the query may not be addressed.""",
        ),
        RubricItem(
            score=4,
            description="""\
The response is largely relevant to the query, addressing the key points without much extraneous \
information. It may not cover all aspects of the query exhaustively.""",
        ),
        RubricItem(
            score=5,
            description="""\
The response is highly relevant to the query, addressing all key aspects directly and thoroughly \
without any irrelevant or extraneous information.""",
        ),
    ],
    required_inputs=["query", "context"],
    required_output="response",
)


def list_all_metrics():
    """List all metric variable names."""
    return [
        name for name, value in globals().items() if isinstance(value, Metric) and name.isupper()
    ]
