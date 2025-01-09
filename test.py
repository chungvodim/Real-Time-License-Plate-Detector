import json

result_texts = [
    '{\'<OCR>\': "OB\'UB4668\\n"}',
    "{'<OCR>': 'UB\\n'}",
    "{'<OCR>': 'B.D.B.1\\n'}",
    "{'<OCR>': 'B\\n'}",
    "{'<OCR>': 'DB\\n'}"
]
for result_text in result_texts:
    key = "'<OCR>': "
    start_index = result_text.find(key) + len(key) + 1
    end_index = result_text.rfind("}") - 1
    value = result_text[start_index:end_index]
    value = value.replace("\\n", "")
    print(value)