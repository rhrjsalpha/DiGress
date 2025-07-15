import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

if __name__ == "__main__":
    # Set stdout encoding to UTF-8 to handle various characters
    sys.stdout.reconfigure(encoding='utf-8')

    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_text.py <pdf_file_path>")
    else:
        pdf_file_path = sys.argv[1]
        extracted_text = extract_text_from_pdf(pdf_file_path)
        print(extracted_text)