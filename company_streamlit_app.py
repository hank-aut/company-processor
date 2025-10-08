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

# Page configuration
st.set_page_config(
    page_title="Company Processor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏢 Company Processor")
    st.markdown("---")

    # Check credentials first
    sf_username = os.getenv("SALESFORCE_USERNAME")
    sf_password = os.getenv("SALESFORCE_PASSWORD")
    sf_token = os.getenv("SALESFORCE_TOKEN")

    # Show credential status
    if not all([sf_username, sf_password, sf_token]):
        st.error("⚠️ Missing Salesforce credentials!")
        st.info("Please configure your environment variables or Streamlit secrets.")

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
        st.info("""
        **What this tool does:**
        • Discovers company website/domain
        • Classifies companies using Salesforce data
        • Applies Rules of Engagement (ROE) qualification
        • 🤖 AI scores relevancy for prospects (1-5 scale)
        • Generates organized Excel output with multiple tabs

        **Output Categories:**
        • Current Customers
        • Open Opportunities
        • Qualified Prospects (+ AI scoring)
        • No Salesforce Match (+ AI scoring)
        • Disqualified - ROE

        **AI Scoring:**
        • 5 = Perfect Fit (DC operators, investors)
        • 4 = Strong Fit (DC ecosystem)
        • 3 = Moderate Fit (Adjacent services)
        • 2 = Weak Fit (Tangential)
        • 1 = No Fit (Not relevant)
        """)

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
                st.success(f"✅ File uploaded successfully! Found {len(df)} companies")

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
                    st.error(f"❌ Missing required column: Company")
                    st.info("Required column: Company (or company, Company Name, etc.)")
                else:
                    st.success("✅ Required column found!")

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
                            st.warning(f"⚠️ Full processing mode will process all {len(df)} companies")

                    # Process button
                    if st.button("🚀 Process Companies", type="primary", use_container_width=True):
                        process_companies(uploaded_file, test_mode)

            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                st.info("Please ensure your file is a valid CSV format.")

    # Display results if they exist in session state (persists across downloads)
    if 'processing_results' in st.session_state:
        st.markdown("---")
        display_results()

    with col2:
        st.header("📊 Status")

        # Processing stats placeholder
        if 'processing_stats' not in st.session_state:
            st.info("Upload a CSV file to begin processing")
        else:
            stats = st.session_state.processing_stats

            # Display metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Total Processed", stats.get('total', 0))
            with col_metric2:
                st.metric("Domains Found", stats.get('domains_found', 0))

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
        with status_placeholder.container():
            st.info("🔄 Initializing processing...")

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

        with status_placeholder.container():
            if test_mode:
                st.info(f"🧪 Test mode: Processing first {processing_limit} of {total_companies} companies")
            else:
                st.info(f"🚀 Full mode: Processing all {total_companies} companies")

        # Initialize progress tracking
        progress_bar = progress_placeholder.progress(0)
        current_processed = 0
        domains_found = 0

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
                    domain_discovery = DomainDiscovery(self.output_dir)
                    domain, domain_notes = domain_discovery.find_domain(company_name)
                except Exception as e:
                    domain = None
                    domain_notes = f"Error during domain discovery: {str(e)}"

                if domain:
                    company_data['domain'] = domain
                    self.domain_count += 1
                    log_entry['domain_discovery']['result'] = 'SUCCESS'
                    log_entry['domain_discovery']['domain_found'] = domain
                    log_entry['domain_discovery']['notes'] = domain_notes
                else:
                    log_entry['domain_discovery']['result'] = 'FAILED'
                    log_entry['domain_discovery']['domain_found'] = None
                    log_entry['domain_discovery']['notes'] = domain_notes

                # Salesforce classification phase with robust error handling
                log_entry['salesforce_classification']['started'] = True
                try:
                    if hasattr(self, 'sf_classifier'):
                        classification, details = self.sf_classifier.classify_company(company_name, domain)
                    else:
                        self.sf_classifier = SalesforceClassifier()
                        classification, details = self.sf_classifier.classify_company(company_name, domain)
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

                # AI Relevancy Scoring - will be done in batch after all companies processed
                # Just initialize the fields for now
                if classification in ['no_salesforce_match', 'salesforce_qualified']:
                    company_data['ai_score'] = 'Pending'
                    company_data['ai_category'] = 'Pending'
                    company_data['ai_use_case'] = 'Pending'
                    company_data['ai_reasoning'] = 'Pending'
                    log_entry['ai_scoring'] = {'started': False, 'pending_batch': True}

                # Final result
                log_entry['final_result']['category'] = classification
                log_entry['final_result']['has_domain'] = domain is not None
                log_entry['final_result']['ready_for_outreach'] = classification in ['salesforce_qualified', 'no_salesforce_match']
                if classification in ['no_salesforce_match', 'salesforce_qualified']:
                    log_entry['final_result']['ai_score'] = company_data.get('ai_score', 'N/A')

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
        def update_progress(current, total, domains_found):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)

            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed", f"{current}/{total}")
                with col2:
                    st.metric("Domains Found", domains_found)
                with col3:
                    completion = f"{progress:.1%}"
                    st.metric("Progress", completion)

        def update_status(message):
            with status_placeholder.container():
                st.info(f"🔄 {message}")

        # Initialize custom processor
        processor = StreamlitCompanyProcessor(
            temp_output_dir,
            progress_callback=update_progress,
            status_callback=update_status
        )

        # Process companies one by one with progress updates
        with status_placeholder.container():
            st.info("🔍 Starting domain discovery and Salesforce classification...")

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

        # Phase 3: Batch AI Relevancy Scoring
        companies_to_score = results['no_salesforce_match'] + results['salesforce_qualified']

        if companies_to_score:
            with status_placeholder.container():
                st.info(f"🤖 AI Scoring Phase: Scoring {len(companies_to_score)} companies in batches...")

            try:
                from ai_relevancy_scorer import AIRelevancyScorer
                ai_scorer = AIRelevancyScorer()

                # Score in batches of 10
                scoring_results = ai_scorer.batch_score_companies(companies_to_score, batch_size=10)

                # Apply scores back to the results
                for classification in ['no_salesforce_match', 'salesforce_qualified']:
                    for company in results[classification]:
                        company_name = company['company']
                        if company_name in scoring_results:
                            ai_result = scoring_results[company_name]
                            company['ai_score'] = ai_result['score']
                            company['ai_category'] = ai_result['category']
                            company['ai_use_case'] = ai_result['use_case']
                            company['ai_reasoning'] = ai_result['reasoning']

                            # Update processing log if available
                            for log_entry in processor.processing_log:
                                if log_entry['company'] == company_name:
                                    log_entry['ai_scoring'] = {
                                        'started': True,
                                        'result': 'SUCCESS',
                                        'score': ai_result['score'],
                                        'category': ai_result['category']
                                    }
                                    log_entry['final_result']['ai_score'] = ai_result['score']
                                    break

                with status_placeholder.container():
                    st.success(f"✅ AI Scoring completed for {len(scoring_results)} companies")

            except Exception as e:
                with status_placeholder.container():
                    st.warning(f"⚠️ AI Scoring encountered an error: {str(e)}")
                # Companies will have 'Pending' scores if AI scoring fails

        # Generate Excel output using existing CSVProcessor
        CSVProcessor.write_results(temp_output_dir, results)

        # Final status update
        with status_placeholder.container():
            st.success("✅ Processing completed successfully!")

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
            st.error("❌ Excel file was not generated. Please check the logs.")

        # Clean up temporary files
        try:
            os.unlink(temp_csv_path)
        except:
            pass

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.info("Please check your CSV format and try again.")

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
    st.info(f"✅ Results generated at: {processed_at}")

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
            st.warning("Excel file no longer available for preview")
    except Exception as e:
        st.warning(f"Preview unavailable: {str(e)}")

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
        if domain_disc.get('result') == 'SUCCESS':
            log_lines.append(f"   ✅ SUCCESS - Domain found: {domain_disc['domain_found']}")
            log_lines.append(f"   Details: {domain_disc.get('notes', 'No details')}")
        else:
            log_lines.append(f"   ❌ FAILED - No domain found")
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

        # AI Scoring Section (if applicable)
        if 'ai_scoring' in entry and entry['ai_scoring'].get('started'):
            log_lines.append("🤖 AI RELEVANCY SCORING:")
            ai_scoring = entry['ai_scoring']
            if ai_scoring.get('result') == 'SUCCESS':
                log_lines.append(f"   ✅ SUCCESS")
                log_lines.append(f"   Score: {ai_scoring.get('score')}/5")
                log_lines.append(f"   Category: {ai_scoring.get('category')}")
            else:
                log_lines.append(f"   ❌ FAILED")
                if ai_scoring.get('error'):
                    log_lines.append(f"   Error: {ai_scoring['error']}")
            log_lines.append("")

        # Final Result Section
        log_lines.append("📊 FINAL RESULT:")
        final = entry['final_result']
        log_lines.append(f"   Final Category: {final['category'].upper()}")
        log_lines.append(f"   Has Domain: {'Yes' if final['has_domain'] else 'No'}")
        log_lines.append(f"   Ready for Outreach: {'Yes' if final['ready_for_outreach'] else 'No'}")
        if 'ai_score' in final and final['ai_score'] != 'N/A':
            log_lines.append(f"   AI Score: {final['ai_score']}/5")
        log_lines.append("")
        log_lines.append("=" * 80)
        log_lines.append("")

    # Summary statistics
    log_lines.append("PROCESSING SUMMARY:")
    log_lines.append("-" * 30)
    total_processed = len(processing_log)
    domains_found = sum(1 for entry in processing_log if entry['final_result']['has_domain'])
    ready_for_outreach = sum(1 for entry in processing_log if entry['final_result']['ready_for_outreach'])

    categories = {}
    for entry in processing_log:
        cat = entry['final_result']['category']
        categories[cat] = categories.get(cat, 0) + 1

    log_lines.append(f"Total Companies Processed: {total_processed}")
    log_lines.append(f"Domains Found: {domains_found} ({domains_found/total_processed*100:.1f}%)")
    log_lines.append(f"Ready for Outreach: {ready_for_outreach} ({ready_for_outreach/total_processed*100:.1f}%)")
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
                    st.info(f"No companies in {sheet_name} category")

    except Exception as e:
        st.error(f"Error previewing results: {str(e)}")

if __name__ == "__main__":
    main()
