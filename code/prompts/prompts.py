CHECKWORTHY_PROMPT = """
Your task is to identify whether a given tweet text in the {lang} language is verifiable using a search engine in the context of fact-checking.
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

CHECKWORTHY_PROMPT_OLLAMA = """
Your task is to identify whether a given tweet text in the {lang} language is verifiable using a search engine in the context of fact-checking.
Let's define a function named checkworthy(input: str).
The return value should be a strings, where each string selects from "Yes", "No".
"Yes" means the text is a factual checkworthy statement.
"No" means that the text is not checkworthy, it might be an opinion, a question, or others.
For example, if a user call checkworthy("I think Apple is a good company.")
You should return a JSON "{{'Label': 'No'}}" without any other words, 
checkworthy("Apple's CEO is Tim Cook.") should return a JSON "{{'Label': 'Yes'}}" since it is verifiable.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!
Always return in JSON format with key "Label" and value "Yes" or "No" without any other words.

checkworthy({text})
"""

CHECKWORTHY_PROMPT_OLLAMA_ML = """
Your task is to identify whether a given tweet text is verifiable using a search engine in the context of fact-checking.
Let's define a function named checkworthy(input: str).
The return value should be a strings, where each string selects from "Yes", "No".
"Yes" means the text is a factual checkworthy statement.
"No" means that the text is not checkworthy, it might be an opinion, a question, or others.
For example, if a user call checkworthy("I think Apple is a good company.")
You should return a JSON "{{'Label': 'No'}}" without any other words, 
checkworthy("Apple's CEO is Tim Cook.") should return a JSON "{{'Label': 'Yes'}}" since it is verifiable.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!
Always return in JSON format with key "Label" and value "Yes" or "No" without any other words.

checkworthy({text})
"""

CHECKWORTHY_PROMPT_MULTILINGUAL = """
Your task is to identify whether a given tweet text is verifiable using a search engine in the context of fact-checking.
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

SUBJ_PROMPT = """
Your task is to identify whether a given text in the {lang} language is subjective or objective.
Let's define a function named objective(input: str).
The return value should be a strings, where each string selects from "OBJ", "SUBJ".
"OBJ" means the text is an objective view of the author behind it.
"SUBJ" means that the text is not objective, it might be a personal opinion or view influenced by personal feelings and experiences.
For example, if a user call objective("Gone are the days when they led the world in recession-busting")
You should return a string "OBJ" without any other words, 
objective("Pakistan has over 7,000 glaciers, the largest number outside the polar region.") should return "OBJ" since it is objective and can be verified.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!
Always return "OBJ" or "SUBJ" without any other words.

objective({text})
"""

SUBJ_PROMPT_ML = """
Your task is to identify whether a given text is subjective or objective.
Let's define a function named objective(input: str).
The return value should be a strings, where each string selects from "OBJ", "SUBJ".
"OBJ" means the text is an objective view of the author behind it.
"SUBJ" means that the text is not objective, it might be a personal opinion or view influenced by personal feelings and experiences.
For example, if a user call objective("Gone are the days when they led the world in recession-busting")
You should return a string "SUBJ" without any other words, 
objective("Pakistan has over 7,000 glaciers, the largest number outside the polar region.") should return "OBJ" since it is objective and can be verified.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!
Always return "OBJ" or "SUBJ" without any other words.

checkworthy({text})
"""

SUBJ_PROMPT_ML_OLLAMA = """
Your task is to identify whether a given text is subjective or objective.
Let's define a function named objective(input: str).
The return value should be a strings, where each string selects from "OBJ", "SUBJ".
"OBJ" means the text is an objective view of the author behind it.
"SUBJ" means that the text is not objective, it might be a personal opinion or view influenced by personal feelings and experiences.
For example, if a user call objective("Gone are the days when they led the world in recession-busting")
You should return a JSON "{{'Label': 'No'}}" without any other words, 
objective("Pakistan has over 7,000 glaciers, the largest number outside the polar region.") 
should return a JSON "{{'Label': 'OBJ'}}" since it is objective and can be verified.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!
Always return "OBJ" or "SUBJ" without any other words.

checkworthy({text})
"""

SUBJ_PROMPT_OLLAMA = """
Your task is to identify whether a given text in the {lang} language is subjective or objective.
Let's define a function named objective(input: str).
The return value should be a strings, where each string selects from "OBJ", "SUBJ".
"OBJ" means the text is an objective view of the author behind it.
"SUBJ" means that the text is not objective, it might be a personal opinion or view influenced by personal feelings and experiences.
For example, if a user call objective("Gone are the days when they led the world in recession-busting")
You should return a JSON "{{'Label': 'No'}}" without any other words, 
objective("Pakistan has over 7,000 glaciers, the largest number outside the polar region.") 
should return a JSON "{{'Label': 'OBJ'}}" since it is objective and can be verified.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!
Always return "OBJ" or "SUBJ" without any other words.

checkworthy({text})
"""

# CHECKWORTHY_PROMPT = """
# مهمتك هي تحديد ما إذا كان نص تغريدة معين يمكن التحقق منه باستخدام محرك بحث في سياق التحقق من الحقائق.
# دعونا نحدد دالة اسمها checkworthy(input: str).
# يجب أن تكون القيمة المرجعة عبارة عن سلاسل، حيث يتم تحديد كل سلسلة من "نعم" و"لا".
# "نعم" تعني أن النص عبارة عن بيان واقعي جدير بالتدقيق.
# "لا" تعني أن النص غير قابل للتدقيق، فقد يكون رأيا أو سؤالا أو غير ذلك.
# على سبيل المثال، إذا قام مستخدم باستدعاء checkworthy("إن من يقبل الصلح مع (اسرائيل) فقد خرج من الإسلام #الشهيد_محمد_باقر_الصدر #لا_لصفقة_القرن #تسقط_صفقة_القرن #القدس_عاصمة_فلسطين_الأبدية ") فيجب عليك إرجاع السلسلة "No" بدون أي كلمات أخرى، checkworthy("مؤسسات طبية روسية تشكك في فعالية لقاح #كورونا (العربية) https://t.co/imiYgy0J16.") يجب عليك إرجاع "Yes" لأنها قابلة للتحقق. لاحظ أنه سيتم تمرير إجابتك إلى مترجم بايثون، لذلك لا توجد كلمات أخرى!
# قم دائمًا بإرجاع "نعم" أو "لا" بدون أي كلمات أخرى.

# checkworthy({text})
# """

CHECKWORTHY_PROMPT_FT = """
Is this sentence fact-check worthy? {text}
"""

IDENTIFY_STANCE_PROMPT = """You are given a claim and an evidence text both in the {lang} language, and you need to decide whether the evidence supports or refutes. Choose from the following two options.
A. The evidence supports the claim. 
B. The evidence refutes the claim.

For example, you are give Claim: "India has the largest population in the world.", Evidence: "In 2023 India overtook China to become the most populous country." You should return A
Pick the correct option either A or B. You must not add any other words.

Claim: {claim}
Evidence: {evidence}"""


IDENTIFY_STANCE_PROMPT_THREE_LABELS = """You are given a claim and an evidence text both in the {lang} language, and you need to decide whether the evidence supports, refutes or not enough info. Choose from the following three options.
A. The evidence supports the claim. 
B. The evidence refutes the claim.
C. The evidence is not related to the claim.

For example, you are give Claim: "India has the largest population in the world.", Evidence: "In 2023 India overtook China to become the most populous country." You should return A
Another example, Claim: "China has the largest population in the world.", Evidence: "In 2023 India overtook China to become the most populous country." You should return B
Another example, Claim: "India has the largest population in the world.", Evidence: "India is the largest democracy in the world." You should return C
Pick the correct option either A, B or C. You must not add any other words.

Claim: {claim}
Evidence: {evidence}"""