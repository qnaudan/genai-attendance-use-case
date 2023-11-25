from io import BytesIO

from src.ocr import process_document
from src.llm import online_predict_with_llm_model


if __name__ == "__main__":
    with open("data/genai_talks.pdf", "rb") as f:
        data = BytesIO(f.read()).getvalue()

    document = process_document(file_bytes=data)

    with open("data/output_ocr.txt", "w") as f:
        f.write(document.text)

    engineered_prompt = """I extracted all the texts from an attendance sheet with an OCR tool.
    This sheet presents a title, a date, and a table with a column for the full names of the participants,
    and a column for their signatures (or mention of absence, or remote attendance through videoconference tool
    such as Teams, Google Meet, and others you may recognize from you market knowledge).

    Some additional information are given in a handwriting way : like additional participants, or any mention
    related to business like display of interest or will to be recontacted.

    It is important to note that OCR returns list of texts and that some were grouped on the same row on original
    document, then it is important to understand how to best group matching elements that belong to a same row.

    From the list of raw texts extracted by the OCR tool, I want you to render a 3 columns table with the following
    information :
    - Column named "Names" with the full names (family name and surname) of the participants you recognize.
    It is mandatory that you specify if any name is uncertain to you and label it with "(uncertain)" keywork in parenthesis.
    - Column "Attendance" with the mention of attendance or absence, the only allowed labels are "signature recognized",
    "remote through videocall", "absent" and "uncertain". It is mandatory that you use the category "uncertain" in case of.
    - Column "Information" with any additional information that is intelligible and useful for my business,
    leave blank if none. It is very important to not append any OCR extracted text that is not intelligible,
    or already used to define the previous columns (like handwritten signatures extracted with OCR or videoconference tool names).

    To sum up, render the data as follows:
    | Names  | Attendance | Information |
    |--------|------------|-------------|
    | [name] |  [status]  |    [info]   |

    Here is the raw text below :
    {text}"""
    response = online_predict_with_llm_model(input_text=document.text, prompt_format=engineered_prompt)

    with open("data/predicted_output", "w") as f:
        f.write(response)
