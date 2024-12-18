import os
import sys
import json
from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

def main():
    # Default directories and artifacts path if not provided via command line arguments
    # input_dir = sys.argv[1] if len(sys.argv) > 1 else "pdfs"
    # output_dir = sys.argv[2] if len(sys.argv) > 2 else "markdowns"
    # artifacts_path = sys.argv[3] if len(sys.argv) > 3 else "./docling_artifacts"
    # input_dir = sys.argv[1] if len(sys.argv) > 1 else "/container/guidelines/pdf"
    # output_dir = sys.argv[2] if len(sys.argv) > 2 else "/container/guidelines/markdown"
    # artifacts_path = sys.argv[3] if len(sys.argv) > 3 else "/container/models/docling_artifacts"
    input_dir = "/container/guidelines/pdf_edited"
    output_dir = "/container/guidelines/markdown2"
    artifacts_path = "/container/models/docling_artifacts"

    # Ensure the artifacts path exists
    # if not os.path.exists(artifacts_path):
    #     os.makedirs(artifacts_path)

    # If you want to prefetch artifacts explicitly, uncomment the line below:
    # artifacts_dl_path = StandardPdfPipeline.download_models_hf()

    # # Move the downloaded artifacts to the specified path
    # if artifacts_dl_path:
    #     for file in os.listdir(artifacts_dl_path):
    #         src = os.path.join(artifacts_dl_path, file)
    #         dst = os.path.join(artifacts_path, file)
    #         os.rename(src, dst)

    # Set up PDF pipeline options
    pipeline_options = PdfPipelineOptions(
        # artifacts_path=artifacts_path,
        do_ocr=False,
        do_table_structure=True,
    )

    # Configure table structure model to use ACCURATE mode and disable cell matching
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    # pipeline_options.table_structure_options.do_cell_matching = False

    # Create the DocumentConverter for PDFs only, with custom pipeline options and backend
    doc_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],  # Only accept PDF inputs
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                # backend=PyPdfiumDocumentBackend  # optional custom backend
            )
        },
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert all PDF files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            try:
                # Convert the single PDF file
                result = doc_converter.convert(pdf_path)
                # Export the document to Markdown (Docling v2 uses result.document)
                md_content = result.document.export_to_markdown()

                # Write to corresponding Markdown file
                base_name = os.path.splitext(filename)[0]
                md_path = os.path.join(output_dir, f"{base_name}.md")
                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(md_content)

                print(f"Converted '{pdf_path}' to '{md_path}'.")
            except Exception as e:
                # If conversion fails, print error and continue with next file
                print(f"Failed to convert '{pdf_path}': {e}")

if __name__ == "__main__":
    main()