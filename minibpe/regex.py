import regex as re 


# Example pattern and string
pattern = r"\d+"  # matches one or more digits
text = "There are 42 apples"

# Test if pattern exists in text
if re.search(pattern, text):
    print("Pattern found!")
else:
    print("Pattern not found.")