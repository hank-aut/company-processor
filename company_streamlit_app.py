#!/usr/bin/env python3
"""
Streamlit Web App for Company Processing
Professional interface for team access to the company workflow
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import io
from datetime import datetime
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Fix SSL certificate issue for Salesforce
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Import our existing processor
from company_processor import CompanyProcessor, CSVProcessor, DomainDiscovery, SalesforceClassifier

# ---------------------------------------------------------------------------
# Visual theme configuration
# ---------------------------------------------------------------------------

PRIMARY_COLOR = "#4E79A7"
BACKGROUND_COLOR = "#0f172a"
SECONDARY_BACKGROUND = "#1e293b"
TEXT_COLOR = "#e2e8f0"

STATUS_STYLES = {
    "success": {
        "bg": "linear-gradient(120deg, rgba(18,50,32,0.9) 0%, rgba(18,50,32,0.45) 100%)",
        "border": "#25A244",
        "icon": "✅",
    },
    "info": {
        "bg": "linear-gradient(120deg, rgba(15,39,68,0.9) 0%, rgba(15,39,68,0.45) 100%)",
        "border": PRIMARY_COLOR,
        "icon": "ℹ️",
    },
    "warning": {
        "bg": "linear-gradient(120deg, rgba(59,39,9,0.9) 0%, rgba(59,39,9,0.4) 100%)",
        "border": "#FAAD14",
        "icon": "⚠️",
    },
    "danger": {
        "bg": "linear-gradient(120deg, rgba(64,12,19,0.9) 0%, rgba(64,12,19,0.45) 100%)",
        "border": "#FF6B6B",
        "icon": "❌",
    },
}


def inject_global_styles():
    """Inject shared CSS for theming tweaks."""
    st.markdown(
        f"""
        <style>
        :root {{
            --primary-color: {PRIMARY_COLOR};
        }}
        .stApp {{
            background: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
            font-family: "Inter", "Segoe UI", sans-serif;
        }}
        section[data-testid="stSidebar"] {{
            background: #0b1120;
            color: {TEXT_COLOR};
        }}
        section[data-testid="stSidebar"] .sidebar-content,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li {{
            color: #cbd5f5;
            font-size: 0.92rem;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {TEXT_COLOR};
        }}
        .sidebar-divider {{
            border: none;
            border-top: 1px solid rgba(148, 163, 184, 0.25);
            margin: 1.5rem 0;
        }}
        div[data-testid="stStatusWidget"] {{
            background: {SECONDARY_BACKGROUND};
        }}
        button[kind="primary"] {{
            background: linear-gradient(135deg, #48689e, {PRIMARY_COLOR});
            border: none;
        }}
        button[kind="primary"]:hover {{
            background: linear-gradient(135deg, #3d5a8a, #41699a);
        }}
        .stMetric label {{
            color: #cbd5f5;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_block(message: str, variant: str = "info", icon: str | None = None, target=None):
    """Render a custom status banner."""
    style = STATUS_STYLES.get(variant, STATUS_STYLES["info"])
    icon = icon or style["icon"]
    render = target.markdown if target is not None else st.markdown
    render(
        f"""
        <div style="
            padding: 0.9rem 1.1rem;
            border-left: 3px solid {style['border']};
            border-radius: 6px;
            background: {style['bg']};
            color: {TEXT_COLOR};
            font-size: 0.95rem;
            margin-bottom: 0.75rem;">
            <span style="margin-right: 0.6rem;">{icon}</span>{message}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Page configuration
st.set_page_config(
    page_title="Company Processor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_global_styles()

def main():
    st.title("🏢 Company Processor")
    st.markdown("---")

    # Check credentials first
    sf_username = os.getenv("SALESFORCE_USERNAME")
    sf_password = os.getenv("SALESFORCE_PASSWORD")
    sf_token = os.getenv("SALESFORCE_TOKEN")

    # Show credential status
    if not all([sf_username, sf_password, sf_token]):
        status_block("Missing Salesforce credentials!", variant="danger")
        status_block("Please configure your environment variables or Streamlit secrets.", variant="info")

        with st.expander("Required Environment Variables"):
            st.code("""
SALESFORCE_USERNAME=your_salesforce_username
SALESFORCE_PASSWORD=your_salesforce_password
SALESFORCE_TOKEN=your_salesforce_security_token
            """)
        return

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **What this tool does:**
        • Discovers company website/domain
        • Classifies companies using Salesforce data
        • Applies Rules of Engagement (ROE) qualification
        • Generates organized Excel output with multiple tabs

        **Output Categories:**
        • Current Customers
        • Open Opportunities
        • Qualified Prospects
        • No Salesforce Match
        • Disqualified - ROE
        """)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        st.header("📋 Instructions")
        st.markdown("""
        1. Upload your company CSV file
        2. Ensure CSV has column: Company
        3. Click 'Process Companies'
        4. Download the Excel results when complete
        """)

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload Company List")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with column: Company"
        )

        if uploaded_file is not None:
            # Preview the uploaded data
            try:
                df = pd.read_csv(uploaded_file)
                status_block(f"File uploaded successfully! Found {len(df)} companies", variant="success")

                # Show preview
                with st.expander("📋 Preview uploaded data", expanded=False):
                    st.dataframe(df.head(10))

                # Validate required columns
                required_columns = ['Company']
                missing_columns = [col for col in required_columns if col not in df.columns]

                # Try alternate column names
                alternate_columns = ['company', 'COMPANY', 'Company Name', 'company_name']
                has_company_column = any(col in df.columns for col in required_columns + alternate_columns)

                if not has_company_column:
                    status_block("Missing required column: Company", variant="danger")
                    status_block("Required column: Company (or company, Company Name, etc.)", variant="info")
                else:
                    status_block("Required column found!", variant="success")

                    # Processing options
                    st.header("⚙️ Processing Options")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        test_mode = st.checkbox(
                            "Test Mode (Process first 50 companies only)",
                            value=True,
                            help="Recommended for testing. Uncheck to process all companies."
                        )

                    with col_b:
                        if not test_mode:
                            status_block(f"Full processing mode will process all {len(df)} companies", variant="warning")

                    # Process button
                    if st.button("🚀 Process Companies", type="primary", use_container_width=True):
                        process_companies(uploaded_file, test_mode)

            except Exception as e:
                status_block(f"Error reading CSV file: {str(e)}", variant="danger")
                status_block("Please ensure your file is a valid CSV format.", variant="info")

    # Display results if they exist in session state (persists across downloads)
    if 'processing_results' in st.session_state:
        st.markdown("---")
        display_results()

    with col2:
        st.header("📊 Status")

        # Processing stats placeholder
        if 'processing_stats' not in st.session_state:
            status_block("Upload a CSV file to begin processing", variant="info")
        else:
            stats = st.session_state.processing_stats

            # Display metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Total Processed", stats.get('total', 0))
            with col_metric2:
                st.metric("Domains Verified", stats.get('domains_found', 0))

            # Progress bar
            if stats.get('total', 0) > 0:
                progress = stats.get('processed', 0) / stats.get('total', 1)
                st.progress(progress)

def process_companies(uploaded_file, test_mode=True):
    """Process the uploaded CSV file using our existing workflow"""

    # Create progress indicators
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()

    try:
        status_block("Initializing processing...", variant="info", icon="🔄", target=status_placeholder)

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_csv_path = tmp_file.name

        # Create temporary output directory
        temp_output_dir = tempfile.mkdtemp()

        # Read the CSV to get total count
        df = pd.read_csv(temp_csv_path)
        total_companies = len(df)
        processing_limit = 50 if test_mode else total_companies

        if test_mode:
            status_block(
                f"Test mode: processing first {processing_limit} of {total_companies} companies",
                variant="info",
                icon="🧪",
                target=status_placeholder,
            )
        else:
            status_block(
                f"Full mode: processing all {total_companies} companies",
                variant="warning",
                icon="⚠️",
                target=status_placeholder,
            )

        # Initialize progress tracking
        progress_bar = progress_placeholder.progress(0)
        current_processed = 0

        # Create custom processor with progress callbacks
        class StreamlitCompanyProcessor(CompanyProcessor):
            def __init__(self, output_dir, progress_callback=None, status_callback=None):
                super().__init__(output_dir)
                self.progress_callback = progress_callback
                self.status_callback = status_callback
                self.current_count = 0
                self.domain_count = 0
                # Limit processing log to prevent memory issues
                self.processing_log = []
                self.batch_size = 50

            def process_company_with_progress(self, company_data, index, total):
                """Process single company with progress updates"""
                from datetime import datetime

                company_name = company_data['company']
                self.current_count = index + 1
                log_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'company_num': self.current_count,
                    'company': company_name,
                    'domain_discovery': {},
                    'salesforce_classification': {},
                    'final_result': {}
                }

                # Update status
                if self.status_callback:
                    self.status_callback(f"Processing {self.current_count}/{total}: {company_name}")

                # Domain discovery phase with error handling
                log_entry['domain_discovery']['started'] = True
                try:
                    domain, domain_verified, domain_notes = self.domain_discovery.find_domain(company_name)
                except Exception as e:
                    domain = None
                    domain_verified = False
                    domain_notes = f"Error during domain discovery: {str(e)}"

                company_data['domain_notes'] = domain_notes
                company_data['domain_verified'] = domain_verified
                log_entry['domain_discovery']['notes'] = domain_notes
                log_entry['domain_discovery']['verified'] = domain_verified

                if domain and domain_verified:
                    company_data['domain'] = domain
                    self.domain_count += 1
                    log_entry['domain_discovery']['result'] = 'VERIFIED'
                    log_entry['domain_discovery']['domain_found'] = domain
                elif domain:
                    company_data['domain_guess'] = domain
                    log_entry['domain_discovery']['result'] = 'UNVERIFIED'
                    log_entry['domain_discovery']['domain_found'] = domain
                else:
                    log_entry['domain_discovery']['result'] = 'FAILED'
                    log_entry['domain_discovery']['domain_found'] = None

                # Salesforce classification phase with robust error handling
                log_entry['salesforce_classification']['started'] = True
                try:
                    if hasattr(self, 'sf_classifier'):
                        classification, details = self.sf_classifier.classify_company(
                            company_name,
                            company_data.get('domain') if company_data.get('domain_verified') else None
                        )
                    else:
                        self.sf_classifier = SalesforceClassifier()
                        classification, details = self.sf_classifier.classify_company(
                            company_name,
                            company_data.get('domain') if company_data.get('domain_verified') else None
                        )
                except Exception as e:
                    classification = 'no_salesforce_match'
                    details = {'classification_reason': f'Salesforce error: {str(e)}'}

                # Log Salesforce results
                log_entry['salesforce_classification']['classification'] = classification
                log_entry['salesforce_classification']['details'] = details
                log_entry['salesforce_classification']['reason'] = details.get('classification_reason', 'No specific reason provided')

                # Add reason and owner information
                if classification == 'excluded':
                    company_data['reason'] = details.get('classification_reason', 'Unknown reason')
                elif classification == 'current_customers':
                    company_data['relationship_owner'] = details.get('account_owner')
                    company_data['account_id'] = details.get('account_id')
                    company_data['account_url'] = details.get('account_url')
                elif classification == 'open_opportunities':
                    company_data['opportunity_owner'] = details.get('opportunity_owner')
                    company_data['opportunity_id'] = details.get('opportunity_id')
                    company_data['opportunity_url'] = details.get('opportunity_url')


                # Final result
                log_entry['final_result']['category'] = classification
                log_entry['final_result']['domain_verified'] = company_data.get('domain_verified', False)
                log_entry['final_result']['ready_for_outreach'] = classification in ['salesforce_qualified', 'no_salesforce_match']

                # Add to processing log (limit size)
                if len(self.processing_log) < 1000:
                    self.processing_log.append(log_entry)
                elif len(self.processing_log) >= 1000:
                    self.processing_log = self.processing_log[-500:] + [log_entry]

                # Update progress
                if self.progress_callback:
                    self.progress_callback(self.current_count, total, self.domain_count)

                # Force garbage collection every 25 companies
                if self.current_count % 25 == 0:
                    import gc
                    gc.collect()

                return classification, domain

        # Progress callback functions
        def update_progress(current, total, verified_domains):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)

            # Persist stats for sidebar display
            st.session_state.processing_stats = {
                'total': total,
                'processed': current,
                'domains_found': verified_domains
            }

            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed", f"{current}/{total}")
                with col2:
                    st.metric("Domains Verified", verified_domains)
                with col3:
                    completion = f"{progress:.1%}"
                    st.metric("Progress", completion)

        def update_status(message):
            status_block(message, variant="info", icon="🔄", target=status_placeholder)

        # Initialize custom processor
        processor = StreamlitCompanyProcessor(
            temp_output_dir,
            progress_callback=update_progress,
            status_callback=update_status
        )

        # Process companies one by one with progress updates
        status_block("Starting domain discovery and Salesforce classification...", variant="info", icon="🔍", target=status_placeholder)

        # Read companies and limit for test mode
        companies = CSVProcessor.read_companies(temp_csv_path)
        companies_to_process = companies[:processing_limit]

        # Initialize results structure
        results = {
            'current_customers': [],
            'open_opportunities': [],
            'salesforce_qualified': [],
            'no_salesforce_match': [],
            'excluded': []
        }

        # Process each company
        total_companies = len(companies_to_process)

        for i, company_data in enumerate(companies_to_process):
            classification, domain = processor.process_company_with_progress(
                company_data, i, total_companies
            )

            # Add to appropriate results category
            if classification in results:
                results[classification].append(company_data)

            # Force garbage collection periodically
            if (i + 1) % 25 == 0:
                import gc
                gc.collect()

        # Generate Excel output using existing CSVProcessor
        CSVProcessor.write_results(temp_output_dir, results)

        # Final status update
        status_block("Processing completed successfully!", variant="success", target=status_placeholder)

        progress_bar.progress(1.0)

        # Find the generated Excel file
        excel_file_path = os.path.join(temp_output_dir, "company_results.xlsx")

        if os.path.exists(excel_file_path):
            # Read Excel file for download
            with open(excel_file_path, 'rb') as f:
                excel_data = f.read()

            # Generate processing log
            processing_log_text = generate_processing_log(processor.processing_log)

            # Store minimal results in session state
            st.session_state.processing_results = {
                'results_summary': {k: len(v) for k, v in results.items()},
                'excel_data': excel_data,
                'processing_log_text': processing_log_text[:10000] + "... (truncated for memory)" if len(processing_log_text) > 10000 else processing_log_text,
                'excel_file_path': excel_file_path,
                'processed_at': datetime.now().strftime('%Y%m%d_%H%M%S')
            }

            # Clear large objects from memory
            del results
            del processing_log_text
            import gc
            gc.collect()

            # Display results
            display_results()

        else:
            status_block("Excel file was not generated. Please check the logs.", variant="danger")

        # Clean up temporary files
        try:
            os.unlink(temp_csv_path)
        except:
            pass

    except Exception as e:
        status_block(f"Error during processing: {str(e)}", variant="danger")
        status_block("Please check your CSV format and try again.", variant="info")

def display_results():
    """Display processing results from session state"""
    if 'processing_results' not in st.session_state:
        return

    stored_results = st.session_state.processing_results
    results_summary = stored_results.get('results_summary', {})
    excel_data = stored_results['excel_data']
    processing_log_text = stored_results['processing_log_text']
    excel_file_path = stored_results['excel_file_path']
    processed_at = stored_results['processed_at']

    st.header("📊 Processing Results")
    status_block(f"Results generated at: {processed_at}", variant="info")

    # Display results summary from counts only
    show_results_summary_from_counts(results_summary)

    # Download buttons in columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.download_button(
            label="📥 Download Excel Results",
            data=excel_data,
            file_name=f"company_results_{processed_at}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )

    with col2:
        st.download_button(
            label="📋 Download Processing Log",
            data=processing_log_text,
            file_name=f"processing_log_{processed_at}.txt",
            mime="text/plain",
            type="secondary",
            use_container_width=True
        )

    with col3:
        # Clear results button
        if st.button("🗑️ Clear Results", use_container_width=True):
            del st.session_state.processing_results
            st.rerun()

    # Show preview of results if Excel file still exists
    try:
        if os.path.exists(excel_file_path):
            show_results_preview(excel_file_path)
        else:
            status_block("Excel file no longer available for preview", variant="warning")
    except Exception as e:
        status_block(f"Preview unavailable: {str(e)}", variant="warning")

def generate_processing_log(processing_log):
    """Generate a human-readable processing log text file"""
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("COMPANY PROCESSING LOG")
    log_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 80)
    log_lines.append("")

    for entry in processing_log:
        log_lines.append(f"COMPANY #{entry['company_num']}: {entry['company']}")
        log_lines.append("-" * 60)
        log_lines.append(f"Processed: {entry['timestamp']}")
        log_lines.append("")

        # Domain Discovery Section
        log_lines.append("🌐 DOMAIN DISCOVERY:")
        domain_disc = entry['domain_discovery']
        domain_result = domain_disc.get('result')
        if domain_result == 'VERIFIED':
            log_lines.append(f"   ✅ VERIFIED - Domain: {domain_disc.get('domain_found', 'Unknown')}")
        elif domain_result == 'UNVERIFIED':
            log_lines.append(f"   ⚠️ UNVERIFIED GUESS - Domain: {domain_disc.get('domain_found', 'Unknown')}")
        else:
            log_lines.append("   ❌ FAILED - No domain found")
        log_lines.append(f"   Details: {domain_disc.get('notes', 'No details')}")
        log_lines.append("")

        # Salesforce Classification Section
        log_lines.append("🏢 SALESFORCE CLASSIFICATION:")
        sf_class = entry['salesforce_classification']
        log_lines.append(f"   Category: {sf_class['classification'].upper()}")
        log_lines.append(f"   Reason: {sf_class['reason']}")
        if sf_class.get('details'):
            details = sf_class['details']
            if details.get('sf_data'):
                sf_data = details['sf_data']
                log_lines.append(f"   Matched Account: {sf_data.get('name', 'Unknown')}")
                if sf_data.get('id'):
                    log_lines.append(f"   Salesforce Account ID: {sf_data['id']}")
                if sf_data.get('customer_designation'):
                    log_lines.append(f"   Customer Status: {sf_data['customer_designation']}")
            if details.get('roe_check'):
                log_lines.append(f"   ROE Check Details: {details['roe_check']}")
        log_lines.append("")


        # Final Result Section
        log_lines.append("📊 FINAL RESULT:")
        final = entry['final_result']
        log_lines.append(f"   Final Category: {final['category'].upper()}")
        log_lines.append(f"   Domain Verified: {'Yes' if final.get('domain_verified') else 'No'}")
        log_lines.append(f"   Ready for Outreach: {'Yes' if final.get('ready_for_outreach') else 'No'}")
        log_lines.append("")
        log_lines.append("=" * 80)
        log_lines.append("")

    # Summary statistics
    log_lines.append("PROCESSING SUMMARY:")
    log_lines.append("-" * 30)
    total_processed = len(processing_log)
    domains_verified = sum(1 for entry in processing_log if entry['final_result'].get('domain_verified'))
    ready_for_outreach = sum(1 for entry in processing_log if entry['final_result'].get('ready_for_outreach'))

    categories = {}
    for entry in processing_log:
        cat = entry['final_result']['category']
        categories[cat] = categories.get(cat, 0) + 1

    log_lines.append(f"Total Companies Processed: {total_processed}")
    if total_processed:
        log_lines.append(f"Domains Verified: {domains_verified} ({domains_verified/total_processed*100:.1f}%)")
        log_lines.append(f"Ready for Outreach: {ready_for_outreach} ({ready_for_outreach/total_processed*100:.1f}%)")
    else:
        log_lines.append("Domains Verified: 0")
        log_lines.append("Ready for Outreach: 0")
    log_lines.append("")
    log_lines.append("Category Breakdown:")
    for category, count in categories.items():
        log_lines.append(f"  {category.title().replace('_', ' ')}: {count}")

    return "\n".join(log_lines)

def show_results_summary_from_counts(results_summary):
    """Display processing results summary from count dictionary"""
    # Map internal keys to display names
    display_mapping = {
        'current_customers': 'Current Customers',
        'open_opportunities': 'Open Opportunities',
        'salesforce_qualified': 'Qualified Prospects',
        'no_salesforce_match': 'No SF Match',
        'excluded': 'Disqualified - ROE'
    }

    # Build totals dictionary
    totals = {}
    for key, display_name in display_mapping.items():
        count = results_summary.get(key, 0)
        totals[display_name] = count

    # Display metrics in columns
    cols = st.columns(len(totals))

    for i, (category, count) in enumerate(totals.items()):
        with cols[i]:
            st.metric(category, count)

    # Total processed
    total_processed = sum(totals.values())
    st.metric("**Total Processed**", total_processed)

def show_results_preview(excel_file_path):
    """Show preview of the Excel results"""
    st.header("👀 Results Preview")

    try:
        # Read all sheets from Excel file
        excel_sheets = pd.read_excel(excel_file_path, sheet_name=None)

        # Create tabs for each sheet
        tab_names = list(excel_sheets.keys())
        tabs = st.tabs(tab_names)

        for i, (sheet_name, df) in enumerate(excel_sheets.items()):
            with tabs[i]:
                if len(df) > 0:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"Total: {len(df)} companies")
                else:
                    status_block(f"No companies in {sheet_name} category", variant="info")

    except Exception as e:
        status_block(f"Error previewing results: {str(e)}", variant="danger")

if __name__ == "__main__":
    main()
