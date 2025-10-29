#!/usr/bin/env python3
"""Fix indentation for inlined Commodity and Crypto sections."""

# Read the file
with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the elif asset_class == "Commodity": line
# Everything from there until the next elif/else at the same indentation level
# needs to be indented by 4 spaces (except the elif line itself)

new_lines = []
in_commodity = False
in_crypto = False
commodity_indent_level = 0
crypto_indent_level = 0

for i, line in enumerate(lines):
    # Check for Commodity start
    if 'elif asset_class == "Commodity":' in line and not in_commodity:
        in_commodity = True
        commodity_indent_level = len(line) - len(line.lstrip())
        new_lines.append(line)
        continue

    # Check for Commodity end (next elif/else at same level, or end of function)
    if in_commodity:
        current_indent = len(line) - len(line.lstrip())
        # If we hit another elif/else/def at the same or lower indent level, commodity section is done
        if line.strip() and (line.strip().startswith('elif ') or line.strip().startswith('else:') or line.strip().startswith('def ')):
            if current_indent <= commodity_indent_level:
                in_commodity = False
                # Check if this is the start of crypto
                if 'elif asset_class == "Crypto":' in line:
                    in_crypto = True
                    crypto_indent_level = current_indent
                new_lines.append(line)
                continue

        # We're still in commodity - indent this line by 4 more spaces
        if line.strip():  # Only indent non-empty lines
            new_lines.append("    " + line)
        else:
            new_lines.append(line)  # Keep empty lines as-is
        continue

    # Check for Crypto end
    if in_crypto:
        current_indent = len(line) - len(line.lstrip())
        if line.strip() and (line.strip().startswith('elif ') or line.strip().startswith('else:') or line.strip().startswith('def ')):
            if current_indent <= crypto_indent_level:
                in_crypto = False
                new_lines.append(line)
                continue

        # We're still in crypto - indent this line by 4 more spaces
        if line.strip():
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
        continue

    # Not in commodity or crypto, just append as-is
    new_lines.append(line)

# Write the fixed file
with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("✅ Fixed indentation for Commodity and Crypto sections!")
print(f"Total lines: {len(new_lines)}")
