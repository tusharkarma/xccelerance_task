import pytesseract
from PIL import Image
import textwrap
import langextract as lx
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Path to tesseract (if not auto-detected)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

@app.route('/extract', methods=['POST'])
def extract_invoice():
    # 1. Get file and prompt from user
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    user_prompt = request.form.get('prompt')

    if not user_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # 2. Read text from image using OCR
    img = Image.open(image_file)
    extracted_text = pytesseract.image_to_string(img)

    # 3. Prepare LangExtract request
    prompt = textwrap.dedent(user_prompt)

    examples =[
        lx.data.ExampleData(
            text="""
        Invoice Number: INV-00123
        Invoice Date: 2025-08-10
        Vendor: ABC Supplies Pvt Ltd
        Total Amount: ₹12,500.00
        Item: Office Chair, Qty: 2, Price: ₹5,000.00, Total: ₹10,000.00
        Item: Delivery Charge, Qty: 1, Price: ₹2,500.00, Total: ₹2,500.00
""",
            extractions=[
                lx.data.Extraction(
                extraction_class="invoice_number",
                extraction_text="INV-00123"
                ),
                lx.data.Extraction(
                    extraction_class="invoice_date",
                    extraction_text="2025-08-10"
                ),
                lx.data.Extraction(
                    extraction_class="vendor_name",
                    extraction_text="ABC Supplies Pvt Ltd"
                ),
                lx.data.Extraction(
                    extraction_class="total_amount",
                    extraction_text="₹12,500.00"
                ),
                lx.data.Extraction(
                    extraction_class="item",
                    extraction_text="Office Chair",
                    attributes={"quantity": "2", "price": "₹5,000.00", "total": "₹10,000.00"}
                ),
                lx.data.Extraction(
                    extraction_class="item",
                    extraction_text="Delivery Charge",
                    attributes={"quantity": "1", "price": "₹2,500.00", "total": "₹2,500.00"}
                ),
            ]
        )
    ]  # You can keep your example data here if you want training context

    # 4. Call langextract API
    result = lx.extract(
        text_or_documents=extracted_text,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.5-pro",
        api_key=api_key
    )

    # 5. Return results in JSON
    output = [
        {
            "class": item.extraction_class,
            "text": item.extraction_text,
            "attributes": item.attributes
        }
        for item in result.extractions
    ]

    return jsonify({
        "extracted_text": extracted_text.strip(),
        "structured_data": output
    })

if __name__ == '__main__':
    app.run(debug=True)
