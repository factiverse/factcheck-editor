from src.llm_utils.ollama import Ollama


if __name__ == "__main__":
    ollama = Ollama()
    # prompt = """Your task is to classify if the given claim has all the context information such as people, location and date needed to verify the claim. If there is no further information needed from the user You MUST generate 'Yes' if there is sufficient context and if user needs to specify further information generate 'No'. Generate nothing else.
    # Here are some examples:
    # Claim: President is a liar.
    # Label: No
    # Claim: Capital of Norway is Oslo.
    # Label: Yes
    # Claim: Earth is flat.
    # Label: Yes
    # Claim: Vaccines are harmful.
    # Label: No
    # Claim: {claim}
    # Label:
    # """
    prompt = """Given a claim, classify whether the claim contains sufficient context information to be verifiable. The output should be either 1 if the claim has all the necessary context, or 0 if the claim is missing key context details.
    Example Claims and Labels.
    Claim: "The President is a liar. Label: 0 (The claim lacks specific context regarding what statement is being referred to, who the President is at the time, and when the statement was made.)
    Claim: "The capital of Norway is Oslo." Label: 1 (The claim provides specific geographical information that can be verified without additional context.)
    Claim: "Earth is flat." Label: 1 (The claim presents a verifiable assertion based on physical geography, regardless of its accuracy.)
    Claim: "Vaccines are harmful. Label: 0 (The claim is too broad and lacks specifics about which vaccines, what type of harm, and under what conditions.)
    Claim: "Oslo is the capital of Norway." Label: 1 (Similar to the previous example, it contains verifiable geographical information.)
    Claim: "Stavanger is the capital of Norway." Label: 1 (It contains verifiable geographical information even though the claim is incorrect.)
    Claim: "Crime rate is decreasing. Label: 0 (The claim lacks specifics about the location, time frame, and types of crime being referred to.)
    Claim: "{claim}"  Label: 
    """
    claim = "Stavanger is the capital of Norway."
    prompt1 = prompt.format(claim=claim)
    print(claim)
    response = ollama.generate(prompt1)
    print(response)
    claim = "Unemployment rate is decreasing."
    prompt2 = prompt.format(claim=claim)
    print(claim)
    response = ollama.generate(prompt2)
    print(response)