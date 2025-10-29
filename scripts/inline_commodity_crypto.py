#!/usr/bin/env python3
"""
Inline show_commodity_technical_analysis() and show_crypto_technical_analysis()
into show_technical_analysis_page() to match the Equity structure.
"""

import re

# Read the file
with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line numbers
commodity_call_line = None
crypto_call_line = None
commodity_func_start = None
commodity_func_end = None
crypto_func_start = None
crypto_func_end = None

for i, line in enumerate(lines):
    if "show_commodity_technical_analysis()" in line and "def " not in line:
        commodity_call_line = i
    elif "show_crypto_technical_analysis()" in line and "def " not in line:
        crypto_call_line = i
    elif line.strip().startswith("def show_commodity_technical_analysis"):
        commodity_func_start = i
    elif line.strip().startswith("def show_crypto_technical_analysis"):
        crypto_func_start = i
        # The commodity function ends just before the crypto function starts
        commodity_func_end = i - 1
        # Trim any blank lines at the end
        while lines[commodity_func_end].strip() == "":
            commodity_func_end -= 1

# Find where crypto function ends - it's the next "def " line
if crypto_func_start is not None:
    for i in range(crypto_func_start + 1, len(lines)):
        if lines[i].strip().startswith("def ") and not lines[i].strip().startswith("def "):
            crypto_func_end = i - 1
            while lines[crypto_func_end].strip() == "":
                crypto_func_end -= 1
            break
    else:
        # If no next function found, go to end
        crypto_func_end = len(lines) - 1

print(f"Commodity call at line {commodity_call_line + 1}")
print(f"Crypto call at line {crypto_call_line + 1}")
print(f"Commodity function: lines {commodity_func_start + 1}-{commodity_func_end + 1}")
print(f"Crypto function: lines {crypto_func_start + 1}-{crypto_func_end + 1}")

# Extract the function bodies (excluding the def line and docstring)
commodity_body = []
crypto_body = []

# For commodity, skip def line and docstring
in_docstring = False
docstring_done = False
for i in range(commodity_func_start + 1, commodity_func_end + 1):
    line = lines[i]
    # Skip docstring
    if not docstring_done:
        if '"""' in line or "'''" in line:
            if in_docstring:
                in_docstring = False
                docstring_done = True
                continue
            else:
                in_docstring = True
                continue
        elif in_docstring:
            continue
    else:
        commodity_body.append(line)

# Same for crypto
in_docstring = False
docstring_done = False
for i in range(crypto_func_start + 1, crypto_func_end + 1):
    line = lines[i]
    if not docstring_done:
        if '"""' in line or "'''" in line:
            if in_docstring:
                in_docstring = False
                docstring_done = True
                continue
            else:
                in_docstring = True
                continue
        elif in_docstring:
            continue
    else:
        crypto_body.append(line)

# Now reconstruct the file:
# 1. Lines before commodity call
# 2. Commodity body (indented)
# 3. Lines between commodity and crypto calls
# 4. Crypto body (indented)
# 5. Lines between crypto call and commodity function start
# 6. Lines after crypto function end

new_lines = []

# Lines before commodity call (including the elif line)
for i in range(commodity_call_line + 1):
    new_lines.append(lines[i])

# Add commodity body (already properly indented from the function)
new_lines.extend(commodity_body)

# Lines between commodity call and crypto call (skip the commodity call line itself)
for i in range(commodity_call_line + 1, crypto_call_line + 1):
    if i != commodity_call_line:
        new_lines.append(lines[i])

# Add crypto body
new_lines.extend(crypto_body)

# Lines from after crypto call to before commodity function definition (skip crypto call line)
for i in range(crypto_call_line + 1, commodity_func_start):
    if i != crypto_call_line:
        new_lines.append(lines[i])

# Lines after crypto function end
new_lines.extend(lines[crypto_func_end + 1:])

# Write the new file
with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

original_lines = len(lines)
new_line_count = len(new_lines)
removed = original_lines - new_line_count

print(f"\n✅ Success!")
print(f"Original: {original_lines} lines")
print(f"New: {new_line_count} lines")
print(f"Removed: {removed} lines (function definitions)")
print("\nCommodity and Crypto are now inlined like Equity!")
