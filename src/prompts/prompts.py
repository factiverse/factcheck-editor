CHECKWORTHY_PROMPT = """
Your task is to identify whether a given text in the {lang} language os verifiable using a search engine in the context of fact-checking.
Let's define a function named checkworthy(input: str).
The return value should be a strings, where each string selects from "Yes", "No".
"Yes" means the text is a factual checkworthy statement.
"No" means that the text is not checkworthy, it might be an opinion, a question, or others.
For example, if a user call checkworthy("I think Apple is a good company.")
You should return a string "No" without any other words, 
checkworthy("Apple's CEO is Tim Cook.") should return "Yes" since it is verifiable.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!
Always return "Yes" or "No" without any other words.

checkworthy({text})
"""

STANCE_SYSTEM_PROMPT = """You are an expert fact-checking assistant. Your task is to determine the stance of a given evidence text in relation to a claim. """

IDENTIFY_STANCE_PROMPT = """Given a claim and it's associated evidence, both in the {lang} language, Your task is to determine the stance of a given evidence text in relation to a claim. Choose from the following.
SUPPORTS: The evidence supports the claim.
REFUTES: The evidence refutes the claim.
MIXED: The evidence contains both supporting and refuting information.

Respond with exactly one of: SUPPORTS, REFUTES, MIXED.

Examples:
Claim: "India has the largest population in the world.", Evidence: "When did India overtake China in population? In 2023 India overtook China to become the most populous country." You should return "SUPPORTS".
Claim: "The earth is flat.", Evidence: "Numerous scientific studies and satellite images have confirmed that the earth is spherical." You should return "REFUTES".
Claim: "The new policy will improve healthcare.", Evidence: "The new policy has some provisions that enhance healthcare access, but it also has budget cuts that may negatively impact services." You should return "MIXED".

Claim: {claim}
Evidence: {evidence}"""