#!/usr/bin/env python3
"""
AI-Powered Relevancy Scoring Module for DataCenterHawk
Uses Claude API to score companies on their fit for DataCenterHawk platform
"""

import os
import json
from typing import Dict, Tuple, Optional
from anthropic import Anthropic

class AIRelevancyScorer:
    def __init__(self):
        """Initialize the AI scorer with Claude API"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=self.api_key)

        # DataCenterHawk ideal customer profile based on current customers
        self.ideal_customer_profile = """
DataCenterHawk is a data center intelligence platform that provides market insights,
facility data, and business intelligence for the data center industry.

**Ideal Customer Types (based on current customers):**

1. **INVESTORS (36% of customers)** - Score: 5
   - Private equity firms investing in data center assets
   - Banks and financial institutions (JP Morgan, Morgan Stanley, UBS)
   - Asset management companies (Brookfield, Stonepeak Partners)
   - Investment advisors (Green Street Advisors)
   - Use Case: Market intelligence for investment decisions

2. **PROVIDERS (25% of customers)** - Score: 5
   - Data center operators (Equinix, Digital Realty, CyrusOne)
   - Colocation providers (Flexential, CoreSite, Vantage)
   - Edge data center operators (EdgeConneX)
   - Cloud infrastructure providers
   - Use Case: Competitive intelligence, market expansion planning

3. **VENDORS (22% of customers)** - Score: 4
   - Data center construction companies (DPR Construction, Hillwood)
   - Critical infrastructure suppliers (power, cooling, fiber)
   - Equipment manufacturers (Corning, Spectrum)
   - Energy companies serving data centers (NextEra Energy)
   - Use Case: Market opportunity identification, sales intelligence

4. **CONSULTANTS (8% of customers)** - Score: 4
   - Strategy consultants (BCG, Deloitte, PwC, EY-Parthenon)
   - Real estate consultants (Ryan LLC, KBC Advisors)
   - Technical advisors for data center projects
   - Use Case: Research and analysis for client projects

5. **BROKERS (4% of customers)** - Score: 4
   - Commercial real estate brokers (CBRE, JLL, Cushman & Wakefield)
   - Specialized data center brokers
   - Use Case: Deal sourcing, market comps

6. **USERS (4% of customers)** - Score: 3-5
   - Hyperscalers with massive DC needs (AWS, Microsoft, NVIDIA)
   - Large enterprises requiring colocation
   - Use Case: Facility selection, vendor evaluation

**NOT a good fit (Score 1-2):**
- Companies with no data center connection
- General IT/software companies (unless DC-focused)
- Retail, consumer goods, unrelated industries
- General construction without DC specialization
"""

    def score_company(self, company_name: str, domain: Optional[str] = None) -> Tuple[int, str, str, str]:
        """
        Score a company's relevancy to DataCenterHawk

        Returns:
            Tuple of (score, category, use_case, reasoning)
            - score: 1-5 relevancy score
            - category: Primary customer type (Investor, Provider, Vendor, etc.)
            - use_case: How they would use DataCenterHawk
            - reasoning: Detailed explanation
        """

        # Build the prompt
        prompt = f"""You are analyzing whether a company would be a good fit for DataCenterHawk, a data center intelligence platform.

{self.ideal_customer_profile}

**Company to Analyze:**
- Name: {company_name}
- Website: {domain if domain else 'Not available'}

**Task:**
Score this company on a 1-5 scale for DataCenterHawk relevancy:
- 5 = Perfect Fit (Data center operators, major investors in DC)
- 4 = Strong Fit (DC ecosystem: construction, vendors, consultants, brokers)
- 3 = Moderate Fit (Adjacent: telecom, cloud, MSPs, users with DC needs)
- 2 = Weak Fit (Tangential connection to data centers)
- 1 = No Fit (No data center industry connection)

**Respond in this exact JSON format:**
{{
    "score": <1-5>,
    "category": "<Investor|Provider|Vendor|Consultant|Broker|User|Not Relevant>",
    "use_case": "<Brief description of how they'd use DataCenterHawk>",
    "reasoning": "<2-3 sentence explanation of the score>"
}}

Be strict - only give high scores to companies clearly in the data center industry ecosystem."""

        try:
            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse the response
            response_text = message.content[0].text

            # Extract JSON from response (in case there's extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]

            result = json.loads(response_text)

            return (
                result.get('score', 1),
                result.get('category', 'Not Relevant'),
                result.get('use_case', 'Unknown'),
                result.get('reasoning', 'No reasoning provided')
            )

        except Exception as e:
            print(f"    AI Scoring Error for '{company_name}': {str(e)}")
            return (1, 'Error', 'Error during scoring', f"Error: {str(e)}")

    def batch_score_companies(self, companies: list) -> Dict[str, Dict]:
        """
        Score multiple companies and return results

        Args:
            companies: List of dicts with 'company' and optional 'domain' keys

        Returns:
            Dict mapping company names to scoring results
        """
        results = {}

        for i, company_data in enumerate(companies):
            company_name = company_data.get('company')
            domain = company_data.get('domain')

            print(f"  Scoring {i+1}/{len(companies)}: {company_name}")

            score, category, use_case, reasoning = self.score_company(company_name, domain)

            results[company_name] = {
                'score': score,
                'category': category,
                'use_case': use_case,
                'reasoning': reasoning,
                'domain': domain
            }

        return results

def test_scorer():
    """Test the AI scorer with sample companies"""
    print("Testing AI Relevancy Scorer...")
    print("="*80)

    scorer = AIRelevancyScorer()

    test_companies = [
        {'company': 'Equinix Inc', 'domain': 'equinix.com'},
        {'company': 'Vertiv', 'domain': 'vertiv.com'},
        {'company': '360 ELECTRIC INC', 'domain': '360electric.com'},
        {'company': 'Walmart', 'domain': 'walmart.com'},
    ]

    results = scorer.batch_score_companies(test_companies)

    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)

    for company, result in results.items():
        print(f"\n{company}")
        print(f"  Score: {result['score']}/5")
        print(f"  Category: {result['category']}")
        print(f"  Use Case: {result['use_case']}")
        print(f"  Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    test_scorer()
