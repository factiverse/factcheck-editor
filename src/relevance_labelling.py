import json
import argparse
import os
from tqdm import tqdm
from src.llm_utils.openai_utils import OpenAIUtils


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_relevance_prompt(claim, evidence_snippet, evidence_title):
    """Create a prompt for the LLM to determine relevance of evidence to claim."""
    prompt = f"""Given the following claim and evidence, determine if the evidence is relevant to verifying the claim.

Claim: {claim}

Evidence Title: {evidence_title}

Evidence Snippet: {evidence_snippet}

Is this evidence relevant for verifying the claim? 

Respond in JSON format with the following fields:
- "relevance": either "RELEVANT" or "NOT_RELEVANT"
- "explanation": a brief one-sentence explanation for your decision

Example response:
{{
  "relevance": "RELEVANT",
  "explanation": "The evidence directly addresses the claim's main point."
}}"""
    
    return prompt


def annotate_evidence_relevance(data, openai_utils, model="gpt-4o", use_json_mode=True, output_file=None):
    """Annotate each evidence item with relevance labels using OpenAI API.
    
    Args:
        data: List of items to annotate
        openai_utils: OpenAI utility instance
        model: Model name to use
        use_json_mode: Whether to use JSON mode for API calls
        output_file: Output file path to write results incrementally
    """
    
    system_message = "You are an expert fact-checker evaluating whether evidence is relevant to verify a claim. Be precise and concise in your judgments. Always respond in valid JSON format."
    
    # Open output file for writing if provided
    output_fh = None
    if output_file:
        output_fh = open(output_file, "w", encoding="utf-8")
    
    try:
        for item in tqdm(data, desc="Annotating evidence"):
            claim = item.get("claim", "")
            
            # Check if factiverse_response exists and has evidence
            if "factiverse_response" not in item:
                # Write item as-is if no evidence to annotate
                if output_fh:
                    output_fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    output_fh.flush()
                continue
                
            factiverse_response = item["factiverse_response"]
            
            if "evidence" not in factiverse_response or not factiverse_response["evidence"]:
                # Write item as-is if no evidence to annotate
                if output_fh:
                    output_fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    output_fh.flush()
                continue
            
            # Annotate each piece of evidence
            for evidence in factiverse_response["evidence"]:
                # Skip if already annotated
                if "relevance_label" in evidence:
                    continue
                
                evidence_snippet = evidence.get("evidenceSnippet", evidence.get("snippet", ""))
                evidence_title = evidence.get("title", "")
                
                # Create prompt for relevance classification
                prompt = create_relevance_prompt(claim, evidence_snippet, evidence_title)
                
                try:
                    # Get response from OpenAI with increased token limit
                    response = openai_utils.generate(
                        prompt, 
                        model=model, 
                        max_tokens=150,
                        system_message=system_message,
                        response_format={"type": "json_object"} if use_json_mode else None
                    )
                    
                    # Parse the JSON response
                    try:
                        # Try to parse as JSON
                        response_json = json.loads(response)
                        evidence["relevance_label"] = response_json.get("relevance", "UNKNOWN")
                        evidence["relevance_explanation"] = response_json.get("explanation", "")
                    except json.JSONDecodeError:
                        # Fallback: try to extract JSON from markdown code blocks
                        if "```json" in response:
                            json_start = response.find("{")
                            json_end = response.rfind("}") + 1
                            if json_start != -1 and json_end > json_start:
                                response_json = json.loads(response[json_start:json_end])
                                evidence["relevance_label"] = response_json.get("relevance", "UNKNOWN")
                                evidence["relevance_explanation"] = response_json.get("explanation", "")
                            else:
                                raise
                        elif "{" in response and "}" in response:
                            # Try to extract JSON from response
                            json_start = response.find("{")
                            json_end = response.rfind("}") + 1
                            response_json = json.loads(response[json_start:json_end])
                            evidence["relevance_label"] = response_json.get("relevance", "UNKNOWN")
                            evidence["relevance_explanation"] = response_json.get("explanation", "")
                        else:
                            # If no JSON found, use old parsing logic as fallback
                            if "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper():
                                evidence["relevance_label"] = "RELEVANT"
                            elif "NOT_RELEVANT" in response.upper():
                                evidence["relevance_label"] = "NOT_RELEVANT"
                            else:
                                evidence["relevance_label"] = "UNKNOWN"
                            evidence["relevance_explanation"] = response
                        
                except Exception as e:
                    print(f"\nError annotating evidence: {e}")
                    print(f"Response was: {response if 'response' in locals() else 'No response'}")
                    evidence["relevance_label"] = "ERROR"
                    evidence["relevance_explanation"] = str(e)
            
            # Write the annotated item to file immediately
            if output_fh:
                output_fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                output_fh.flush()
    
    finally:
        # Close the output file if it was opened
        if output_fh:
            output_fh.close()
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Annotate evidence with relevance labels')
    parser.add_argument('--input-file', type=str, required=True, 
                        help='Input JSONL file path')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output JSONL file path (default: input_file with _relevance_annotated suffix)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='OpenAI model to use (default: gpt-4o)')
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        base_name = args.input_file.rsplit('.', 1)[0]
        args.output_file = f"{base_name}_relevance_annotated.jsonl"
    
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} items")
    
    # Initialize OpenAI Utils
    print(f"Initializing OpenAI with model: {args.model}")
    openai_utils = OpenAIUtils()
    
    # Annotate evidence with relevance labels
    print("Annotating evidence with relevance labels...")
    annotated_data = annotate_evidence_relevance(
        data, 
        openai_utils, 
        model=args.model,
        output_file=args.output_file
    )
    
    print(f"\nAnnotation complete! Output saved to {args.output_file}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_evidence = 0
    relevant_count = 0
    not_relevant_count = 0
    
    for item in annotated_data:
        if "factiverse_response" in item and "evidence" in item["factiverse_response"]:
            for evidence in item["factiverse_response"]["evidence"]:
                if "relevance_label" in evidence:
                    total_evidence += 1
                    if evidence["relevance_label"] == "RELEVANT":
                        relevant_count += 1
                    elif evidence["relevance_label"] == "NOT_RELEVANT":
                        not_relevant_count += 1
    
    if total_evidence > 0:
        print(f"Total evidence items: {total_evidence}")
        print(f"Relevant: {relevant_count} ({relevant_count/total_evidence*100:.2f}%)")
        print(f"Not Relevant: {not_relevant_count} ({not_relevant_count/total_evidence*100:.2f}%)")
    else:
        print("No evidence items found to annotate.")
    print(f"Not Relevant: {not_relevant_count} ({not_relevant_count/total_evidence*100:.2f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
