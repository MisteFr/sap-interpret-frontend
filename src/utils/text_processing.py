import re

def extract_relevant_text(input_text):
    """Extract the relevant part of the text between 'Human:' and '[/INST]'."""
    if input_text is None:
        return ""
    match = re.search(r"Human:(.*?)\[/INST\]", input_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return input_text.strip()

def clean_token(token):
    """Remove spaces, 'Ġ', 'Ċ', and extra spaces from tokens."""
    if token is None:
        return ""
    token_str = str(token)
    cleaned = re.sub(r'\s+', ' ', token_str.replace('Ġ', '').replace('Ċ', '')).strip()
    return cleaned

def clean_text(input_text):
    """Remove unwanted characters like 'Ċ' and extra spaces from the text."""
    if input_text is None:
        return ""
    return re.sub(r'\s+', ' ', input_text.replace('Ċ', '')).strip() 