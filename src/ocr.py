import os
from typing import Optional

from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

load_dotenv()


def process_document(
    file_bytes: bytes,
    project_id: str = os.environ.get("PROJECT_ID"),
    location: str = "eu",
    processor_id: str = os.environ.get("PROCESSOR_ID"),
    processor_version: str = os.environ.get("PROCESSOR_VERSION"),
    mime_type: str = "application/pdf",
    process_options: Optional[documentai.ProcessOptions] = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_native_pdf_parsing=True,
            enable_image_quality_scores=True,
            enable_symbol=True,
            # OCR Add Ons https://cloud.google.com/document-ai/docs/ocr-add-ons
            premium_features=documentai.OcrConfig.PremiumFeatures(
                compute_style_info=True,
                enable_math_ocr=False,  # Enable to use Math OCR Model
                enable_selection_mark_detection=True,
            ),
        )
    ),
) -> documentai.Document:
    # You must set the `api_endpoint` if you use a location other than "us".
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"
        )
    )
    # The full resource name of the processor version, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
    # You must create a processor before running this sample.
    name = client.processor_version_path(
        project_id, location, processor_id, processor_version
    )
    # Configure the process request
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=file_bytes, mime_type=mime_type),
        # Only supported for Document OCR processor
        process_options=process_options,
    )
    result = client.process_document(request=request)
    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    return result.document