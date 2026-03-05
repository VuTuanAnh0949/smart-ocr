"""
Web UI module for the OCR application using Streamlit
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import io
import logging
import time
from PIL import Image
import base64
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

# Configure Streamlit page FIRST before any other Streamlit commands
st.set_page_config(
    page_title="AI OCR Pro - Vũ Tuấn Anh",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/VuTuanAnh0949',
        'Report a bug': "https://github.com/VuTuanAnh0949/OCR-Image-to-text/issues",
        'About': "# AI OCR Pro\nDeveloped by Vũ Tuấn Anh\nvutuananh0949@gmail.com"
    }
)

# Add parent directory to path for absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ocr_app.core.ocr_engine import OCREngine
from ocr_app.core.image_processor import ImageProcessor
from ocr_app.models.model_manager import ModelManager
from ocr_app.rag.rag_processor import RAGProcessor
from ocr_app.config.settings import Settings
from ocr_app.utils.text_utils import extract_entities, format_ocr_result

logger = logging.getLogger(__name__)

class StreamlitApp:
    """
    Streamlit web interface for the OCR application
    """
    
    def __init__(self):
        """Initialize the Streamlit app"""
        self.settings = Settings()
        self.init_session_state()
        self.load_resources()
    
    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'extracted_text' not in st.session_state:
            st.session_state['extracted_text'] = ''
        
        if 'ocr_engine' not in st.session_state:
            st.session_state['ocr_engine'] = 'auto'
        
        if 'preserve_layout' not in st.session_state:
            st.session_state['preserve_layout'] = True
        
        if 'dependency_errors' not in st.session_state:
            st.session_state['dependency_errors'] = []
            
        if 'models_initialized' not in st.session_state:
            st.session_state['models_initialized'] = False
    
    def load_resources(self):
        """Load required resources and check dependencies"""
        try:
            # Initialize components
            with st.spinner('Initializing OCR system...'):
                self.model_manager = ModelManager(self.settings)
                self.ocr_engine = OCREngine(self.settings)
                self.image_processor = ImageProcessor(self.settings)
                self.rag_processor = RAGProcessor(self.model_manager, self.settings)
                
                # Store available OCR engines in session state
                st.session_state['available_ocr_engines'] = self.ocr_engine.enabled_engines
                
                # Check for missing OCR engines
                all_engines = ['tesseract', 'easyocr', 'paddleocr']
                missing_engines = [engine for engine in all_engines if engine not in self.ocr_engine.enabled_engines]
                st.session_state['missing_ocr_engines'] = missing_engines
                
                # Generate installation instructions for missing engines
                if missing_engines:
                    instructions = []
                    if 'tesseract' in missing_engines:
                        if sys.platform == 'win32':
                            instructions.append("Download and install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
                        elif sys.platform == 'darwin':  # macOS
                            instructions.append("brew install tesseract")
                        else:  # Linux
                            instructions.append("sudo apt-get install tesseract-ocr")
                        instructions.append("pip install pytesseract")
                    
                    if 'easyocr' in missing_engines:
                        instructions.append("pip install easyocr")
                    
                    if 'paddleocr' in missing_engines:
                        instructions.append("pip install paddlepaddle paddleocr")
                    
                    st.session_state['ocr_installation_instructions'] = instructions
                else:
                    st.session_state['ocr_installation_instructions'] = []
                
                # Check module status
                module_status = self.model_manager.get_module_status()
                if not module_status.get('transformers_available', False) or not module_status.get('sentence_transformers_available', False):
                    st.session_state['rag_available'] = False
                    st.session_state['dependency_errors'].append("Q&A functionality is limited - transformers or sentence_transformers modules not available")
                else:
                    st.session_state['rag_available'] = True
                
                st.session_state['models_initialized'] = True
                
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            st.session_state['dependency_errors'].append(f"Error initializing components: {str(e)}")
            st.session_state['models_initialized'] = False
    
    def load_css(self):
        """Load custom CSS styles with modern dark theme"""
        css_file = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'styles.css')
        
        try:
            with open(css_file, encoding='utf-8') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
                logger.info("Custom CSS loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load custom CSS: {e}")
    
    def run(self):
        """Run the Streamlit application"""
        # Load CSS styles
        self.load_css()
        
        # Display header
        self._display_header()
        
        # Sidebar info
        with st.sidebar:
            st.markdown("### Quick Settings")
            available_engines = st.session_state.get('available_ocr_engines', [])
            if available_engines:
                st.markdown(f'<div class="status-badge status-success">{len(available_engines)} Engines Active</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("""
            ### Tips
            - **Auto**: Smart engine selection
            - **Combined**: Best accuracy  
            - **Preserve Layout**: Keep formatting
            - **PDF**: Multi-page support
            
            ### Features
            - Vietnamese Support
            - Word Export (.docx)
            - Math Formula Detection
            - Image Enhancement
            - Multi-page PDF
            """)
        
        # Create main tabs
        tabs = st.tabs(["Document Processing", "OCR Settings", "Q&A Interface"])
        
        # Document Processing Tab
        with tabs[0]:
            self._document_processing_tab()
        
        # OCR Settings Tab
        with tabs[1]:
            self._ocr_settings_tab()
        
        # Q&A Interface Tab
        with tabs[2]:
            self._qa_interface_tab()
    
    def _display_header(self):
        """Display modern animated header"""
        st.markdown("""
        <div class="main-header animated-fade">
            <h1>AI OCR Pro</h1>
            <p>Intelligent Text Extraction with Advanced AI</p>
            <p>Vietnamese Support • Word Export • Math Formulas</p>
            <p style="margin-top: 1.5rem;">
                <a href="https://github.com/VuTuanAnh0949" target="_blank">Vũ Tuấn Anh</a>
                <a href="mailto:vutuananh0949@gmail.com">Contact</a>
                <a href="https://github.com/VuTuanAnh0949/OCR-Image-to-text" target="_blank">GitHub</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _document_processing_tab(self):
        """Handle document processing tab"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload Document")
            uploaded_file = st.file_uploader(
                "Choose an image or PDF file",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"],
                help="Supported formats: JPG, PNG, BMP, TIFF, PDF"
            )
            
            if uploaded_file:
                # Check if uploaded file is PDF
                is_pdf = uploaded_file.name.lower().endswith('.pdf')
                
                if is_pdf:
                    # Handle PDF file
                    st.info("PDF file detected. Processing all pages...")
                    
                    try:
                        import fitz  # PyMuPDF
                        
                        # Open PDF
                        pdf_bytes = uploaded_file.read()
                        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                        
                        # Show PDF info
                        st.markdown(f"**Pages:** {len(pdf_document)}")
                        
                        # Convert first page to image for preview
                        page = pdf_document[0]
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        
                        from io import BytesIO
                        image = Image.open(BytesIO(img_bytes))
                        # Save preview image for Word export
                        st.session_state['original_image'] = image
                        st.session_state['is_pdf'] = True
                        st.session_state['pdf_page_count'] = len(pdf_document)
                        st.image(image, caption=f"PDF Preview (Page 1/{len(pdf_document)})", use_column_width=True)
                        
                        # OCR Processing for PDF
                        if st.button("Extract Text from PDF", type="primary"):
                            with st.spinner(f'Processing {len(pdf_document)} page(s)...'):
                                try:
                                    all_text = []
                                    pdf_pages_data = []  # Store page data for Word export
                                    progress_bar = st.progress(0)
                                    
                                    for page_num in range(len(pdf_document)):
                                        # Update progress
                                        progress_bar.progress((page_num + 1) / len(pdf_document))
                                        
                                        # Get page
                                        page = pdf_document[page_num]
                                        pix = page.get_pixmap()
                                        img_bytes = pix.tobytes("png")
                                        page_image = Image.open(BytesIO(img_bytes))
                                        
                                        # Perform OCR
                                        page_text = self.ocr_engine.perform_ocr(
                                            page_image,
                                            engine=st.session_state.get('ocr_engine', 'auto'),
                                            preserve_layout=st.session_state.get('preserve_layout', True),
                                            preprocess=True
                                        )
                                        
                                        # Store for display
                                        all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                                        
                                        # Store for Word export (format expected by word_exporter)
                                        pdf_pages_data.append({
                                            'image': page_image,
                                            'text': page_text
                                        })
                                    
                                    extracted_text = "\n\n".join(all_text)
                                    
                                    if extracted_text:
                                        st.session_state['extracted_text'] = extracted_text
                                        st.session_state['pdf_pages_data'] = pdf_pages_data  # Save structured data
                                        st.success(f"Text extracted from {len(pdf_document)} page(s)!")
                                    else:
                                        st.error("No text could be extracted from the PDF.")
                                    
                                    progress_bar.empty()
                                    pdf_document.close()
                                    
                                except Exception as e:
                                    st.error(f"Error during PDF OCR processing: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    
                    except ImportError:
                        st.error("PyMuPDF (fitz) is required for PDF processing.")
                        st.code("pip install PyMuPDF")
                    except Exception as e:
                        st.error(f"Error opening PDF: {str(e)}")
                
                else:
                    # Handle image file
                    image = Image.open(uploaded_file)
                    # Save to session state for Word export
                    st.session_state['original_image'] = image
                    st.session_state['is_pdf'] = False
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Image analysis
                    quality_info = self.image_processor.assess_image_quality(image)
                    has_tables = self.image_processor.detect_tables(image)
                    
                    st.markdown("### Image Analysis")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Quality Score", f"{quality_info.get('quality_score', 0):.2f}")
                    with col_b:
                        st.metric("Contains Tables", "Yes" if has_tables else "No")
                    
                    if has_tables:
                        st.info("Table structures detected. Layout preservation is recommended.")
                    
                    # OCR Processing
                    if st.button("Extract Text", type="primary"):
                        with st.spinner('Processing image...'):
                            try:
                                extracted_text = self.ocr_engine.perform_ocr(
                                    image,
                                    engine=st.session_state.get('ocr_engine', 'auto'),
                                    preserve_layout=st.session_state.get('preserve_layout', True),
                                    preprocess=True
                                )
                                
                                if extracted_text:
                                    st.session_state['extracted_text'] = extracted_text
                                    st.success("Text extracted successfully!")
                                else:
                                    st.error("No text could be extracted from the image.")
                                    
                            except Exception as e:
                                st.error(f"Error during OCR processing: {str(e)}")
        
        with col2:
            st.markdown("### Extracted Text")
            
            if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
                text = st.session_state['extracted_text']
                
                # Display text in expandable container
                with st.container():
                    st.text_area(
                        "Extracted text:",
                        value=text,
                        height=400,
                        help="You can copy this text or use it for Q&A"
                    )
                
                # Text statistics
                st.markdown("### Text Statistics")
                word_count = len(text.split())
                char_count = len(text)
                line_count = len(text.split('\n'))
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Words", word_count)
                with col_stats2:
                    st.metric("Characters", char_count)
                with col_stats3:
                    st.metric("Lines", line_count)
                
                # Extract entities
                entities = extract_entities(text)
                if any(entities.values()):
                    st.markdown("### Detected Entities")
                    for entity_type, values in entities.items():
                        if values:
                            st.markdown(f"**{entity_type.title()}**: {', '.join(values[:5])}")
                
                # Download options
                st.markdown("### Download")
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                with col_dl1:
                    st.download_button(
                        label="Download as TXT",
                        data=text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                with col_dl2:
                    formatted_text = format_ocr_result(text, 'markdown')
                    st.download_button(
                        label="Download as Markdown",
                        data=formatted_text,
                        file_name="extracted_text.md",
                        mime="text/markdown"
                    )
                with col_dl3:
                    # Export to Word with format preservation
                    try:
                        from ocr_app.utils.word_exporter import create_word_from_ocr_result, create_word_from_pdf_ocr
                        from datetime import datetime
                        
                        # Get original image(s) if available
                        original_image = st.session_state.get('original_image', None)
                        is_pdf = st.session_state.get('is_pdf', False)
                        pdf_pages_data = st.session_state.get('pdf_pages_data', [])
                        
                        if original_image:
                            metadata = {
                                'Tác giả': 'Vũ Tuấn Anh',
                                'Ngày tạo': datetime.now().strftime('%d/%m/%Y %H:%M'),
                                'OCR Engine': st.session_state.get('ocr_engine', 'auto')
                            }
                            
                            # Use appropriate function based on file type
                            if is_pdf and pdf_pages_data:
                                word_bytes = create_word_from_pdf_ocr(
                                    pdf_pages_data,
                                    metadata=metadata
                                )
                            else:
                                word_bytes = create_word_from_ocr_result(
                                    original_image,
                                    text,
                                    preserve_layout=st.session_state.get('preserve_layout', True),
                                    metadata=metadata
                                )
                            
                            st.download_button(
                                label="Download as Word",
                                data=word_bytes,
                                file_name="extracted_text.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                        else:
                            st.download_button(
                                label="Download as Word",
                                data=text,
                                file_name="extracted_text.txt",
                                mime="text/plain",
                                disabled=True,
                                help="Only available after extracting text"
                            )
                    except Exception as e:
                        st.error(f"Error creating Word file: {str(e)}")
            else:
                st.info("Upload an image and click 'Extract Text' to see results here.")
    
    def _ocr_settings_tab(self):
        """Handle OCR settings tab"""
        st.markdown("### ⚙️ OCR Engine Settings")
        
        # Engine selection
        available_engines = st.session_state.get('available_ocr_engines', [])
        engine_options = ['auto'] + available_engines + ['combined']
        
        st.session_state['ocr_engine'] = st.selectbox(
            "Select OCR Engine",
            options=engine_options,
            index=0,
            help="Choose the OCR engine or let the system decide automatically"
        )
        
        # Layout preservation option
        st.session_state['preserve_layout'] = st.checkbox(
            "Preserve text layout and formatting",
            value=True,
            help="Maintains the original document's layout including line breaks and text positioning"
        )
        
        # Advanced settings
        st.markdown("### Advanced Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Image Preprocessing")
            preprocess_enabled = st.checkbox(
                "Enable image preprocessing",
                value=True,
                help="Applies various image enhancement techniques before OCR"
            )
            
            if preprocess_enabled:
                st.checkbox("Enhance contrast", value=True)
                st.checkbox("Remove noise", value=True)
                st.checkbox("Correct skew", value=True)
        
        with col2:
            st.markdown("#### Performance")
            st.slider("Processing timeout (seconds)", 30, 300, 60)
            st.checkbox("Use GPU acceleration (if available)", value=False)
        
        # Engine comparison
        st.markdown("---")
        st.markdown("### OCR Engine Comparison")
        
        engine_info = {
            "Tesseract": {
                "description": "Open-source OCR engine with good general accuracy",
                "strengths": "Clear typography, well-formatted documents",
                "best_for": "Scanned documents, books, articles"
            },
            "PaddleOCR": {
                "description": "Fast and accurate OCR optimized for multiple languages",
                "strengths": "Speed, multi-language support, handwriting",
                "best_for": "Forms, receipts, mixed content"
            },
            "EasyOCR": {
                "description": "General-purpose OCR with 80+ language support",
                "strengths": "Wide language support, good for various fonts",
                "best_for": "International documents, signage"
            }
        }
        
        for engine, info in engine_info.items():
            if engine.lower() in [e.lower() for e in available_engines]:
                with st.expander(f"{engine}"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Strengths:** {info['strengths']}")
                    st.write(f"**Best for:** {info['best_for']}")
    
    def _qa_interface_tab(self):
        """Handle Q&A interface tab"""
        if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
            st.markdown("### Ask Questions About Your Document")
            
            # Pre-defined question templates
            st.markdown("#### Quick Questions")
            quick_questions = [
                "What is the main topic?",
                "What are the key dates mentioned?",
                "Who are the people mentioned?",
                "What numbers or amounts are mentioned?",
                "Summarize the content",
                "What is the document type?"
            ]
            
            selected_question = st.selectbox("Select a quick question:", ["Custom question..."] + quick_questions)
            
            if selected_question != "Custom question...":
                query = selected_question
            else:
                query = st.text_input("Enter your custom question:")
            
            if query and st.button("Get Answer"):
                with st.spinner('Analyzing document and finding answer...'):
                    try:
                        result = self.rag_processor.process_query(st.session_state['extracted_text'], query)
                        
                        # Display answer
                        st.markdown("### Answer")
                        st.markdown(f"**{result['answer']}**")
                        
                        # Show confidence
                        confidence = result.get('confidence', 0) * 100
                        st.progress(min(confidence / 100, 1.0))
                        st.caption(f"Confidence: {confidence:.1f}%")
                        
                        # Show source passages
                        if 'chunks_used' in result and result['chunks_used']:
                            with st.expander("View source passages"):
                                for i, chunk in enumerate(result['chunks_used']):
                                    score = result.get('chunk_scores', [0])[i] if i < len(result.get('chunk_scores', [])) else 0
                                    st.markdown(f"**Passage {i+1}** (relevance: {score:.2f})")
                                    st.markdown(f"> {chunk}")
                                    
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        st.info("Try rephrasing your question or check if the answer might be in the document.")
            
            # Tips for better questions
            with st.expander("Tips for Better Questions"):
                st.markdown("""
                - Be specific in your questions
                - Ask about information that's likely to be in the document
                - Use keywords that might appear in the text
                - For dates: "What date..." or "When..."
                - For people: "Who..." or "What person..."
                - For numbers: "How much..." or "What amount..."
                - For summaries: "What is the main point..." or "Summarize..."
                """)
                
        else:
            st.info("Please upload and process a document first to use the Q&A feature.")
            
            st.markdown("""
            ### About the Q&A Feature
            
            This feature uses advanced AI to answer questions about your documents:
            
            - **Retrieval-Augmented Generation (RAG)**: Finds relevant parts of your document
            - **Question Answering**: Uses AI models to provide precise answers
            - **Confidence Scoring**: Shows how confident the system is in its answer
            - **Source Attribution**: Shows which parts of the document were used
            
            The system works best with clear, well-formatted documents and specific questions.
            """)

def main():
    """Main entry point for the Streamlit app"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
