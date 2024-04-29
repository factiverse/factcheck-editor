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

IDENTIFY_STANCE_PROMPT = """You are given a claim and an evidence text both in the {lang} language, and you need to decide whether the evidence supports or refutes. Choose from the following two options.
A. The evidence supports the claim. 
B. The evidence refutes the claim.

For example, you are give Claim: "India has the largest population in the world.", Evidence: "In 2023 India overtook China to become the most populous country." You should return A
Pick the correct option either A or B. You must not add any other words.

Claim: {claim}
Evidence: {evidence}"""