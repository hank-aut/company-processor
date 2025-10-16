#!/usr/bin/env python3
"""
Company-Based Processing System
Handles company domain discovery and Salesforce classification without email discovery
"""

import json
import csv
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import time
import requests
import pandas as pd

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Salesforce integration
try:
    from simple_salesforce import Salesforce
    SALESFORCE_AVAILABLE = True
except ImportError:
    SALESFORCE_AVAILABLE = False
    print("Warning: simple-salesforce not installed. Install with: pip install simple-salesforce")

class ProgressTracker:
    def __init__(self, progress_dir: str):
        self.progress_dir = progress_dir
        self.progress_file = os.path.join(progress_dir, 'workflow_progress.json')
        self.ensure_directory()

    def ensure_directory(self):
        os.makedirs(self.progress_dir, exist_ok=True)

    def save_progress(self, data: Dict):
        """Save progress data with timestamp"""
        data['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_progress(self) -> Dict:
        """Load existing progress or return empty state"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'total_companies': 0,
            'processed_count': 0,
            'phase': 'not_started',
            'domain_stats': {'found': 0, 'not_found': 0},
            'sf_stats': {'qualified': 0, 'disqualified': 0, 'no_match': 0, 'current_customer': 0, 'open_opportunity': 0}
        }

class DateCalculator:
    @staticmethod
    def get_today() -> datetime:
        """Get current date for consistent calculations"""
        return datetime.now().date()

    @staticmethod
    def get_cutoff_dates() -> Tuple[datetime, datetime]:
        """Get ROE cutoff dates"""
        today = DateCalculator.get_today()
        activity_cutoff = today - timedelta(days=90)  # LastActivityDate threshold
        system_cutoff = today - timedelta(days=30)    # SystemModstamp threshold
        return activity_cutoff, system_cutoff

    @staticmethod
    def check_roe_qualification(last_activity_str: str, system_modstamp_str: str) -> Tuple[bool, str]:
        """
        Precise ROE qualification check with detailed reasoning
        Returns: (qualified: bool, reason: str)
        """
        try:
            activity_cutoff, system_cutoff = DateCalculator.get_cutoff_dates()

            # Parse dates
            if last_activity_str:
                last_activity = datetime.strptime(last_activity_str.split('T')[0], '%Y-%m-%d').date()
            else:
                last_activity = datetime(1900, 1, 1).date()  # Very old date if None

            system_modstamp = datetime.strptime(system_modstamp_str.split('T')[0], '%Y-%m-%d').date()

            # Calculate days difference for logging
            activity_days = (DateCalculator.get_today() - last_activity).days
            system_days = (DateCalculator.get_today() - system_modstamp).days

            # ROE checks
            activity_pass = last_activity <= activity_cutoff
            system_pass = system_modstamp <= system_cutoff

            if activity_pass and system_pass:
                return True, f"QUALIFIED - Activity: {activity_days}d ago (>{90}d), System: {system_days}d ago (>{30}d)"
            elif not activity_pass:
                return False, f"DISQUALIFIED - Recent activity: {activity_days}d ago (<{90}d threshold)"
            else:
                return False, f"DISQUALIFIED - Recent system update: {system_days}d ago (<{30}d threshold)"

        except Exception as e:
            return False, f"DATE_PARSE_ERROR: {str(e)}"

class CSVProcessor:
    @staticmethod
    def read_companies(file_path: str) -> List[Dict]:
        """Read companies from CSV file"""
        companies = []
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                companies.append({
                    'company': row.get('Company', row.get('company', '')).strip()
                })
        return companies

    @staticmethod
    def write_results(output_dir: str, results: Dict[str, List[Dict]]):
        """Write classification results to a single Excel file with multiple tabs"""
        os.makedirs(output_dir, exist_ok=True)

        # Create Excel file path
        excel_filepath = os.path.join(output_dir, "company_results.xlsx")

        # Create Excel writer object
        with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:

            # Tab names mapping for cleaner display
            tab_names = {
                'current_customers': 'Current Customers',
                'open_opportunities': 'Open Opportunities',
                'salesforce_qualified': 'Qualified Prospects',
                'no_salesforce_match': 'No SF Match',
                'excluded': 'Disqualified - ROE'
            }

            for classification, companies in results.items():
                # Convert companies to DataFrame
                if companies:
                    # Convert to records format for DataFrame
                    records = []
                    for company in companies:
                        record = {
                            'Company': company['company'],
                            'Domain': company.get('domain', 'NOT_FOUND')
                        }

                        # Add Reason column for excluded companies
                        if classification == 'excluded' and 'reason' in company:
                            record['Reason'] = company['reason']

                        # Add Relationship Owner, Account ID, and URL columns for current customers
                        if classification == 'current_customers':
                            if 'relationship_owner' in company:
                                record['Relationship Owner'] = company.get('relationship_owner', '')
                            if 'account_id' in company:
                                record['Account ID'] = company.get('account_id', '')
                            if 'account_url' in company:
                                record['Account URL'] = company.get('account_url', '')

                        # Add Opportunity Owner, ID, and URL columns for open opportunities
                        if classification == 'open_opportunities':
                            if 'opportunity_owner' in company:
                                record['Opportunity Owner'] = company.get('opportunity_owner', '')
                            if 'opportunity_id' in company:
                                record['Opportunity ID'] = company.get('opportunity_id', '')
                            if 'opportunity_url' in company:
                                record['Opportunity URL'] = company.get('opportunity_url', '')


                        records.append(record)
                    df = pd.DataFrame(records)
                else:
                    # Create empty DataFrame with headers
                    if classification == 'excluded':
                        df = pd.DataFrame(columns=['Company', 'Domain', 'Reason'])
                    elif classification == 'current_customers':
                        df = pd.DataFrame(columns=['Company', 'Domain', 'Relationship Owner', 'Account ID', 'Account URL'])
                    elif classification == 'open_opportunities':
                        df = pd.DataFrame(columns=['Company', 'Domain', 'Opportunity Owner', 'Opportunity ID', 'Opportunity URL'])
                    else:
                        df = pd.DataFrame(columns=['Company', 'Domain'])

                # Get clean tab name
                tab_name = tab_names.get(classification, classification.replace('_', ' ').title())

                # Write to Excel tab
                df.to_excel(writer, sheet_name=tab_name, index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets[tab_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"📊 Results saved to: {excel_filepath}")

class DomainDiscovery:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def find_domain(self, company: str) -> Tuple[Optional[str], str]:
        """
        Find company domain/website using Google search
        Returns: (domain or None, detailed_notes)
        """
        notes = []

        # Step 1: Try to extract domain from company name (if it looks like a domain)
        if '.' in company and ' ' not in company:
            # Might be a domain already
            potential_domain = company.lower().strip()
            if potential_domain.startswith('www.'):
                potential_domain = potential_domain[4:]
            notes.append(f"Company name appears to be a domain: {potential_domain}")
            return potential_domain, "; ".join(notes)

        # Step 2: Google search for company website
        print(f"    Google Search: Searching for '{company}' website...")
        domain, search_notes = self.search_google_for_domain(company)
        notes.append(search_notes)

        if domain:
            return domain, "; ".join(notes)

        # Step 3: Try common domain patterns
        print(f"    Pattern Inference: Trying common domain patterns...")
        domain, pattern_notes = self.infer_domain_pattern(company)
        notes.append(pattern_notes)

        return domain, "; ".join(notes)

    def search_google_for_domain(self, company: str) -> Tuple[Optional[str], str]:
        """
        Use Google search to find company website
        Returns: (domain or None, detailed_notes)
        """
        try:
            # Clean company name for search
            clean_company = company.strip()

            # Search query
            query = f'"{clean_company}" company website'

            # Use a simple web search approach (this can be enhanced with actual Google API)
            # For now, we'll try to extract domain from company name patterns

            # Try to construct likely domain
            company_clean = clean_company.lower()
            company_clean = re.sub(r'[^\w\s-]', '', company_clean)  # Remove special chars
            company_clean = company_clean.replace(' ', '')  # Remove spaces

            # Remove common suffixes
            suffixes = ['inc', 'llc', 'corp', 'ltd', 'company', 'co', 'group']
            for suffix in suffixes:
                if company_clean.endswith(suffix):
                    company_clean = company_clean[:-len(suffix)]

            # Try common TLDs
            for tld in ['.com', '.net', '.org', '.io']:
                potential_domain = f"{company_clean}{tld}"
                # In a real implementation, we would verify this domain exists
                # For now, we'll return .com as most likely
                if tld == '.com':
                    return potential_domain, f"Inferred domain: {potential_domain}"

            return None, f"Could not find domain for '{company}'"

        except Exception as e:
            return None, f"Google Search error: {str(e)}"

    def infer_domain_pattern(self, company: str) -> Tuple[Optional[str], str]:
        """
        Try common domain patterns based on company name
        """
        try:
            # Clean company name
            clean = company.lower().strip()
            clean = re.sub(r'[^\w\s-]', '', clean)
            clean = clean.replace(' ', '')

            # Remove common business suffixes
            suffixes = ['inc', 'llc', 'corp', 'ltd', 'company', 'co', 'group', 'companies']
            for suffix in suffixes:
                if clean.endswith(suffix):
                    clean = clean[:-len(suffix)]

            # Try .com first as it's most common
            domain = f"{clean}.com"
            return domain, f"Pattern inference: {domain}"

        except Exception as e:
            return None, f"Pattern inference error: {str(e)}"

class SalesforceClassifier:
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or "/tmp"
        os.makedirs(self.log_dir, exist_ok=True)
        self.sf = None
        # Company-level classification cache for consistency
        self.company_classifications = {}
        self._connect_to_salesforce()

    def _connect_to_salesforce(self):
        """Connect to Salesforce using credentials from environment variables"""
        if not SALESFORCE_AVAILABLE:
            print("    Salesforce: simple-salesforce not available, using simulation")
            return

        try:
            # Get credentials from environment variables
            sf_username = os.getenv('SALESFORCE_USERNAME')
            sf_password = os.getenv('SALESFORCE_PASSWORD')
            sf_token = os.getenv('SALESFORCE_TOKEN')

            if not all([sf_username, sf_password, sf_token]):
                print("    Salesforce: Missing credentials in environment variables")
                print("    Required: SALESFORCE_USERNAME, SALESFORCE_PASSWORD, SALESFORCE_TOKEN")
                self.sf = None
                return

            self.sf = Salesforce(
                username=sf_username,
                password=sf_password,
                security_token=sf_token
            )
            print("    Salesforce: Connected successfully")
        except Exception as e:
            print(f"    Salesforce: Connection failed - {str(e)}")
            self.sf = None

    def search_by_domain(self, domain: str) -> Optional[Dict]:
        """Search Salesforce Accounts by domain"""
        if not self.sf or not domain:
            return None

        try:
            # SOQL search for Accounts with matching domain in Website field
            soql_query = f"SELECT Id, Name, Website, Customer_Designation__c, Owner.Name, LastActivityDate, SystemModstamp FROM Account WHERE Website LIKE '%{domain}%'"

            print(f"    Salesforce: Searching for domain '{domain}' in Account websites...")
            result = self.sf.query(soql_query)

            if result.get('records'):
                # Return first account match
                record = result['records'][0]
                print(f"    Salesforce: Found domain match '{record.get('Name')}' with website containing '{domain}'")
                return {
                    'type': 'Account',
                    'id': record['Id'],
                    'name': record.get('Name'),
                    'website': record.get('Website'),
                    'customer_designation': record.get('Customer_Designation__c'),
                    'account_owner': record.get('Owner', {}).get('Name') if record.get('Owner') else None,
                    'last_activity_date': record.get('LastActivityDate'),
                    'system_modstamp': record.get('SystemModstamp')
                }
            else:
                print(f"    Salesforce: No Account found with domain '{domain}' in website")
                return None

        except Exception as e:
            print(f"    Salesforce: Domain search error - {str(e)}")
            return None

    def search_by_company(self, company: str) -> Optional[Dict]:
        """
        Search Salesforce for company/account using strict fuzzy matching
        Uses similarity scoring to ensure matches are actually the same company
        """
        if not self.sf:
            return None

        try:
            # Normalize the input company name
            normalized_input = self.normalize_company_name(company)

            # Extract the most distinctive word(s) for the SOQL search
            # This ensures we search for meaningful terms, not just generic words
            search_terms = self.get_search_terms(normalized_input)

            if not search_terms:
                print(f"    Salesforce: Could not extract meaningful search terms from '{company}'")
                return None

            # Build SOQL query using the most distinctive terms
            search_term = search_terms[0]  # Use the first/best term
            variation_escaped = search_term.replace("'", "\\'")
            soql_query = f"SELECT Id, Name, Website, Customer_Designation__c, Owner.Name, LastActivityDate, SystemModstamp FROM Account WHERE Name LIKE '%{variation_escaped}%' LIMIT 10"

            print(f"    Salesforce: Searching for '{search_term}' (from '{normalized_input}')...")
            result = self.sf.query(soql_query)

            if result.get('records'):
                # Score each match and pick the best one
                best_match = None
                best_score = 0

                for record in result['records']:
                    sf_name = record.get('Name', '')
                    sf_normalized = self.normalize_company_name(sf_name)

                    # Calculate similarity score
                    score = self.calculate_similarity(normalized_input, sf_normalized)

                    print(f"      Candidate: '{sf_name}' (normalized: '{sf_normalized}') - Score: {score:.2f}")

                    if score > best_score:
                        best_score = score
                        best_match = record

                # Dynamic threshold based on company name characteristics
                # If the main distinctive word matches exactly, accept lower threshold
                threshold = 0.7

                # Check if key words match exactly
                input_words = set(normalized_input.lower().split())
                sf_words = set(self.normalize_company_name(best_match.get('Name', '')).lower().split())

                # Get distinctive words (non-generic)
                distinctive_input = self.get_search_terms(normalized_input)
                distinctive_sf = self.get_search_terms(self.normalize_company_name(best_match.get('Name', '')))

                # If the main distinctive word is shared, lower threshold
                if distinctive_input and distinctive_sf:
                    if distinctive_input[0] in [w.lower() for w in distinctive_sf]:
                        threshold = 0.3  # "Aligned" matches "Aligned Energy" or "Aligned Data Centers"
                        print(f"      Lowered threshold to {threshold} (key word '{distinctive_input[0]}' matches)")

                if best_score >= threshold:
                    print(f"    Salesforce: ✅ Accepted match '{best_match.get('Name')}' with score {best_score:.2f} (threshold: {threshold})")
                    return {
                        'type': 'Account',
                        'id': best_match['Id'],
                        'name': best_match.get('Name'),
                        'website': best_match.get('Website'),
                        'customer_designation': best_match.get('Customer_Designation__c'),
                        'account_owner': best_match.get('Owner', {}).get('Name') if best_match.get('Owner') else None,
                        'last_activity_date': best_match.get('LastActivityDate'),
                        'system_modstamp': best_match.get('SystemModstamp')
                    }
                else:
                    print(f"    Salesforce: ❌ Best match score {best_score:.2f} below threshold ({threshold})")

            print(f"    Salesforce: No acceptable match found for '{company}'")
            return None

        except Exception as e:
            print(f"    Salesforce: Company search error - {str(e)}")
            return None

    def get_search_terms(self, company: str) -> List[str]:
        """
        Extract the most distinctive words from a company name for searching
        Prioritizes unique identifiers over generic industry terms
        """
        words = company.lower().split()

        # Generic words to deprioritize (but not completely exclude)
        filler_words = {
            'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with',
            'inc', 'llc', 'corp', 'ltd', 'co', 'company', 'companies', 'corporation',
            'limited', 'incorporated', 'partners', 'partnership'
        }

        # Words that are TOO generic to search by (would match too many companies)
        too_generic = {
            'power', 'energy', 'development', 'construction', 'infrastructure',
            'solutions', 'services', 'consulting', 'management'
        }

        # Get non-generic words first
        distinctive_words = [w for w in words if w not in filler_words and w not in too_generic and len(w) >= 3]

        # If we found distinctive words, return them
        if distinctive_words:
            return distinctive_words

        # If only filler words remain, use all words except business entity types
        # This handles cases like "Aligned Data Centers" where data/centers ARE the identity
        fallback_words = [w for w in words if w not in filler_words and len(w) >= 3]

        if fallback_words:
            return fallback_words

        # Last resort - use the original company name
        return [company.lower()]

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity score between two company names (0.0 to 1.0)
        Uses multiple metrics:
        - Exact match = 1.0
        - Character overlap ratio
        - Word overlap ratio
        """
        str1_lower = str1.lower().strip()
        str2_lower = str2.lower().strip()

        # Exact match
        if str1_lower == str2_lower:
            return 1.0

        # One contains the other exactly
        if str1_lower in str2_lower or str2_lower in str1_lower:
            # Calculate how much of the shorter string is in the longer
            shorter = min(len(str1_lower), len(str2_lower))
            longer = max(len(str1_lower), len(str2_lower))
            return shorter / longer

        # Word-level matching (most important for company names)
        words1 = set(str1_lower.split())
        words2 = set(str2_lower.split())

        # Skip very generic words and data center industry filler words
        generic_words = {
            # Basic generic words
            'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with',
            # Business entity types
            'inc', 'llc', 'corp', 'ltd', 'co', 'group', 'company', 'companies', 'corporation',
            'limited', 'incorporated', 'partners', 'partnership',
            # Data center industry filler words
            'power', 'energy', 'natural', 'renewable', 'solar', 'wind',
            'development', 'developments', 'developer', 'developers',
            'construction', 'infrastructure', 'engineering', 'design',
            'solutions', 'services', 'service', 'technologies', 'technology', 'tech',
            'systems', 'system', 'consulting', 'consultants', 'consultant',
            'management', 'ventures', 'ventures', 'holdings', 'capital',
            'real', 'estate', 'realestate', 'properties', 'property',
            'electric', 'electrical', 'mechanical', 'industrial',
            'global', 'international', 'national', 'regional', 'local',
            'american', 'north', 'south', 'east', 'west',
            'facility', 'facilities'
        }

        # Filter out generic words, but keep "data" and "center/centers" if they're part of the core identity
        # Only filter them if there are OTHER meaningful words present
        filtered_words1 = {w for w in words1 if w not in generic_words}
        filtered_words2 = {w for w in words2 if w not in generic_words}

        # If filtering removed ALL words, keep the original (don't filter data/center in this case)
        if not filtered_words1 and words1:
            filtered_words1 = {w for w in words1 if w not in {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with', 'inc', 'llc', 'corp', 'ltd', 'co', 'company', 'companies', 'corporation', 'limited', 'incorporated'}}
        if not filtered_words2 and words2:
            filtered_words2 = {w for w in words2 if w not in {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with', 'inc', 'llc', 'corp', 'ltd', 'co', 'company', 'companies', 'corporation', 'limited', 'incorporated'}}

        words1 = filtered_words1
        words2 = filtered_words2

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        word_similarity = intersection / union if union > 0 else 0.0

        # Boost score if all words from input are present in SF name
        if words1.issubset(words2):
            word_similarity = max(word_similarity, 0.85)

        return word_similarity

    def generate_company_variations(self, company: str) -> List[str]:
        """Generate comprehensive company name variations for better matching"""
        variations = []

        # Step 1: Clean and normalize the company name
        clean_company = self.normalize_company_name(company)
        variations.append(company)  # Original
        if clean_company != company:
            variations.append(clean_company)  # Cleaned version

        # Use cleaned version for word splitting to avoid punctuation issues
        words = clean_company.split()

        # Step 2: Generate word combinations
        if len(words) >= 2:
            # First 2 words
            variations.append(' '.join(words[:2]))

            # First and last word
            if len(words) >= 3:
                variations.append(f"{words[0]} {words[-1]}")

            # Last 2 words
            if len(words) >= 3:
                variations.append(' '.join(words[-2:]))

        # Step 3: Individual important words (3+ chars, not common/generic)
        skip_words = {
            'by', 'and', 'the', 'of', 'for', 'with', 'at', 'in', 'on',
            'inc', 'llc', 'corp', 'ltd', 'co', 'group', 'company', 'companies',
            'energy', 'power', 'data', 'center', 'centers', 'solutions', 'services'
        }

        for word in words:
            if len(word) >= 3 and word.lower() not in skip_words:
                variations.append(word)

        # Remove duplicates while preserving order
        unique_variations = []
        seen = set()
        for variation in variations:
            if variation and variation not in seen:
                unique_variations.append(variation)
                seen.add(variation)

        print(f"    Generated variations for '{company}': {unique_variations}")
        return unique_variations

    def normalize_company_name(self, company: str) -> str:
        """Normalize company name by removing common suffixes and cleaning punctuation"""
        suffixes_to_remove = [
            'Inc.', 'Inc', 'LLC.', 'LLC', 'Corp.', 'Corp', 'Ltd.', 'Ltd',
            'Co.', 'Co', 'Group', 'Companies', 'Company'
        ]

        normalized = company.strip()

        for suffix in suffixes_to_remove:
            patterns = [f', {suffix}', f' {suffix}', f',{suffix}']
            for pattern in patterns:
                if normalized.lower().endswith(pattern.lower()):
                    normalized = normalized[:-len(pattern)].strip()
                    break

        normalized = normalized.replace(',', '').replace('.', '').strip()
        normalized = ' '.join(normalized.split())

        return normalized

    def check_open_opportunities(self, account_id: str) -> Tuple[int, Optional[str], Optional[str], Optional[str]]:
        """Check for open opportunities on an account
        Returns: (count, opportunity_owner_name, opportunity_id, opportunity_url)
        """
        if not self.sf:
            return 0, None, None, None

        try:
            soql_query = f"SELECT Id, Name, Owner.Name FROM Opportunity WHERE AccountId = '{account_id}' AND IsClosed = false LIMIT 1"

            print(f"    Salesforce: Checking open opportunities for account {account_id}...")
            result = self.sf.query(soql_query)

            count_query = f"SELECT COUNT() FROM Opportunity WHERE AccountId = '{account_id}' AND IsClosed = false"
            count_result = self.sf.query(count_query)
            count = count_result.get('totalSize', 0)

            owner_name = None
            opportunity_id = None
            opportunity_url = None

            if result.get('records') and len(result['records']) > 0:
                first_opp = result['records'][0]
                owner_name = first_opp.get('Owner', {}).get('Name') if first_opp.get('Owner') else None
                opportunity_id = first_opp.get('Id')

                if opportunity_id:
                    opportunity_url = f"https://datacenterhawk.lightning.force.com/lightning/r/Opportunity/{opportunity_id}/view"

            if count > 0 and (not opportunity_id or not opportunity_url):
                count = 0
                opportunity_id = None
                opportunity_url = None
                owner_name = None

            return count, owner_name, opportunity_id, opportunity_url

        except Exception as e:
            print(f"    Salesforce: Opportunity check error - {str(e)}")
            return 0, None, None, None

    def classify_company(self, company_name: str, domain: Optional[str]) -> Tuple[str, Dict]:
        """
        Classify company based on Salesforce relationship
        Returns: (classification, details)
        """
        details = {'search_results': [], 'classification_reason': '', 'sf_data': None}

        # Check cache first
        if company_name in self.company_classifications:
            cached_classification, cached_details = self.company_classifications[company_name]
            details.update(cached_details)
            details['classification_reason'] = f"Company-level cached: {cached_details.get('classification_reason', 'Unknown reason')}"
            print(f"    Using cached classification for '{company_name}': {cached_classification}")
            return cached_classification, details

        # Search Salesforce using domain first (if available), then company name
        sf_match = None

        if domain:
            print(f"    Searching Salesforce by domain: {domain}")
            sf_match = self.search_by_domain(domain)

        if not sf_match:
            print(f"    Searching Salesforce by company name: {company_name}")
            sf_match = self.search_by_company(company_name)

        if not sf_match:
            classification = 'no_salesforce_match'
            reason = 'No Salesforce relationship found for company'
            details['classification_reason'] = reason
            self.company_classifications[company_name] = (classification, details.copy())
            return classification, details

        # Company has SF relationship
        details['sf_data'] = sf_match

        # Check if current customer
        customer_designation = sf_match.get('customer_designation')

        if customer_designation == 'Current Customer':
            classification = 'current_customers'
            reason = f"Company '{company_name}' is a current customer"
            details['classification_reason'] = reason
            details['account_owner'] = sf_match.get('account_owner')

            account_id = sf_match.get('id')
            if account_id:
                details['account_id'] = account_id
                details['account_url'] = f"https://datacenterhawk.lightning.force.com/lightning/r/Account/{account_id}/view"

            self.company_classifications[company_name] = (classification, details.copy())
            return classification, details

        # Check for open opportunities
        account_id = sf_match.get('id')
        if account_id:
            open_opps, opportunity_owner, opportunity_id, opportunity_url = self.check_open_opportunities(account_id)
            if open_opps > 0:
                classification = 'open_opportunities'
                reason = f"Company has {open_opps} open opportunities"
                details['classification_reason'] = reason
                details['opportunity_owner'] = opportunity_owner
                details['opportunity_id'] = opportunity_id
                details['opportunity_url'] = opportunity_url
                self.company_classifications[company_name] = (classification, details.copy())
                return classification, details

        # ROE qualification check
        last_activity = sf_match.get('last_activity_date')
        system_modstamp = sf_match.get('system_modstamp')

        if last_activity or system_modstamp:
            qualified, roe_reason = DateCalculator.check_roe_qualification(last_activity, system_modstamp)
            details['roe_check'] = roe_reason

            if qualified:
                classification = 'salesforce_qualified'
                reason = f"ROE qualified: {roe_reason}"
            else:
                classification = 'excluded'
                reason = f"ROE disqualified: {roe_reason}"
        else:
            classification = 'salesforce_qualified'
            reason = 'Salesforce match found with no recent activity data - assuming qualified'

        details['classification_reason'] = reason
        self.company_classifications[company_name] = (classification, details.copy())
        return classification, details

class CompanyProcessor:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        self.base_dir = base_dir
        self.progress_tracker = ProgressTracker(os.path.join(base_dir, 'progress'))
        self.domain_discovery = DomainDiscovery(os.path.join(base_dir, 'logs'))
        self.sf_classifier = SalesforceClassifier(os.path.join(base_dir, 'logs'))
        self.output_dir = os.path.join(base_dir, 'output')

    def process_file(self, csv_file_path: str):
        """Process a company CSV file"""
        print(f"Starting company workflow for: {csv_file_path}")

        # Load companies
        companies = CSVProcessor.read_companies(csv_file_path)
        total_count = len(companies)

        print(f"Loaded {total_count} companies")

        # Initialize progress
        progress = self.progress_tracker.load_progress()
        progress.update({
            'total_companies': total_count,
            'processed_count': 0,
            'phase': 'domain_discovery',
            'current_company': None,
            'start_time': datetime.now().isoformat()
        })
        self.progress_tracker.save_progress(progress)

        # Results storage
        results = {
            'current_customers': [],
            'open_opportunities': [],
            'salesforce_qualified': [],
            'no_salesforce_match': [],
            'excluded': []
        }

        # Process each company
        for i, company_data in enumerate(companies):
            company_name = company_data['company']
            print(f"\nProcessing {i+1}/{total_count}: {company_name}")

            # Update progress
            progress['current_company'] = company_name
            progress['processed_count'] = i
            progress['phase'] = 'domain_discovery'
            self.progress_tracker.save_progress(progress)

            # Phase 1: Domain Discovery
            print("  Phase 1: Domain Discovery...")
            domain, domain_notes = self.domain_discovery.find_domain(company_name)

            if domain:
                print(f"    ✅ Domain found: {domain}")
                progress['domain_stats']['found'] += 1
                company_data['domain'] = domain
            else:
                print("    ❌ No domain found")
                progress['domain_stats']['not_found'] += 1

            # Phase 2: Salesforce Classification
            progress['phase'] = 'salesforce_classification'
            self.progress_tracker.save_progress(progress)

            print("  Phase 2: Salesforce Classification...")
            classification, details = self.sf_classifier.classify_company(company_name, domain)

            print(f"    📊 Classification: {classification}")
            print(f"    📝 Reason: {details['classification_reason']}")

            # Update stats
            if classification == 'excluded':
                progress['sf_stats']['disqualified'] += 1
            elif classification == 'no_salesforce_match':
                progress['sf_stats']['no_match'] += 1
            elif classification == 'current_customers':
                progress['sf_stats']['current_customer'] += 1
            elif classification == 'open_opportunities':
                progress['sf_stats']['open_opportunity'] += 1
            elif classification == 'salesforce_qualified':
                progress['sf_stats']['qualified'] += 1

            # Store result
            company_with_details = company_data.copy()

            if classification == 'excluded':
                company_with_details['reason'] = details.get('classification_reason', 'Unknown reason')
            elif classification == 'current_customers':
                company_with_details['relationship_owner'] = details.get('account_owner')
                company_with_details['account_id'] = details.get('account_id')
                company_with_details['account_url'] = details.get('account_url')
            elif classification == 'open_opportunities':
                company_with_details['opportunity_owner'] = details.get('opportunity_owner')
                company_with_details['opportunity_id'] = details.get('opportunity_id')
                company_with_details['opportunity_url'] = details.get('opportunity_url')

            results[classification].append(company_with_details)

        # Final phase: Generate outputs
        print(f"\nGenerating output files...")
        progress['phase'] = 'generating_outputs'
        progress['processed_count'] = total_count
        self.progress_tracker.save_progress(progress)

        CSVProcessor.write_results(self.output_dir, results)

        # Complete
        progress['phase'] = 'completed'
        progress['end_time'] = datetime.now().isoformat()
        self.progress_tracker.save_progress(progress)

        # Print summary
        print(f"\n🎉 Processing Complete!")
        print(f"🌐 Domain Discovery: {progress['domain_stats']['found']} found, {progress['domain_stats']['not_found']} not found")
        print(f"🏢 Salesforce Classification:")
        print(f"   Current Customers: {progress['sf_stats']['current_customer']}")
        print(f"   Open Opportunities: {progress['sf_stats']['open_opportunity']}")
        print(f"   Qualified Prospects: {progress['sf_stats']['qualified']}")
        print(f"   No SF Match: {progress['sf_stats']['no_match']}")
        print(f"   Disqualified: {progress['sf_stats']['disqualified']}")
        print(f"\n📁 Output files saved to: {self.output_dir}")

        return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python company_processor.py <csv_file_path>")
        print("\nExpected CSV format:")
        print("  Company")
        print("  Example Company Inc")
        print("  Another Corp")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    processor = CompanyProcessor(base_dir="/Users/hankautrey/Company Processor")
    processor.process_file(csv_file_path)

if __name__ == "__main__":
    main()
