# Company Processor

A secure company processing system that discovers company domains and classifies companies using Salesforce data with Rules of Engagement (ROE) qualification.

## Features

- 🌐 **Domain Discovery**: Automatically finds company websites/domains
- 🔍 **Salesforce Classification**: Uses domain and exact company name matching
- 📊 **ROE Qualification**: Checks for current customers, open opportunities, and recent activity
- 🤖 **AI Relevancy Scoring**: Claude AI scores "No SF Match" and "Qualified Prospects" on 1-5 scale
- 📥 **Excel Output**: Organized results in multiple tabs with AI insights
- 🖥️ **Web Interface**: Streamlit app for easy team access

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual Salesforce credentials:

```
SALESFORCE_USERNAME=your_salesforce_username
SALESFORCE_PASSWORD=your_salesforce_password
SALESFORCE_TOKEN=your_salesforce_security_token
```

**⚠️ IMPORTANT**: Never commit the `.env` file to Git. It's already in `.gitignore`.

### 3. Get Your Anthropic API Key (for AI Scoring)

1. Go to https://console.anthropic.com/
2. Sign in or create an account
3. Navigate to "API Keys"
4. Create a new API key
5. Add it to your `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

## Usage

### Option 1: Command Line

```bash
python company_processor.py companies.csv
```

**Expected CSV format:**
```csv
Company
Example Company Inc
Another Corp
Third Company LLC
```

### Option 2: Web Interface (Streamlit)

```bash
streamlit run company_streamlit_app.py
```

Then open your browser to the URL shown (usually `http://localhost:8501`)

## Output

The processor generates an Excel file with multiple tabs:

- **Current Customers**: Companies with Customer_Designation__c = "Current Customer"
- **Open Opportunities**: Companies with open deals in Salesforce
- **Qualified Prospects**: Companies that pass ROE qualification checks (+ AI scoring)
- **No SF Match**: Companies not found in Salesforce (+ AI scoring)
- **Disqualified - ROE**: Companies that fail ROE checks (recent activity)

### AI Scoring Columns (for Qualified Prospects & No SF Match)

- **AI Score**: 1-5 relevancy rating for DataCenterHawk platform
- **AI Category**: Customer type (Provider, Investor, Vendor, Consultant, Broker, User)
- **AI Use Case**: How they would use DataCenterHawk
- **AI Reasoning**: Detailed explanation of the score

**Scoring Scale:**
- **5** = Perfect Fit (Data center operators, major DC investors)
- **4** = Strong Fit (DC ecosystem: construction, vendors, consultants, brokers)
- **3** = Moderate Fit (Adjacent: telecom, cloud, MSPs with DC needs)
- **2** = Weak Fit (Tangential connection to data centers)
- **1** = No Fit (No data center industry connection)

## ROE Qualification Rules

A company is qualified if:
- LastActivityDate > 90 days ago
- SystemModstamp > 30 days ago
- Not a current customer
- No open opportunities

## Security

This project uses environment variables to store sensitive credentials. Never commit:
- `.env` file
- Any files containing actual credentials
- API keys or tokens

## Project Structure

```
Company Processor/
├── company_processor.py          # Main processing script
├── company_streamlit_app.py      # Streamlit web interface
├── requirements.txt              # Python dependencies
├── .env.example                  # Example environment variables
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── output/                       # Generated Excel files (gitignored)
├── logs/                         # Processing logs (gitignored)
└── progress/                     # Progress tracking (gitignored)
```

## Troubleshooting

### Salesforce Connection Issues

If you see "Salesforce: Connection failed", verify:
1. Your `.env` file exists and has the correct credentials
2. Your Salesforce password is correct
3. Your security token is current (may need to reset in Salesforce)

### SSL Certificate Issues

If you encounter SSL errors, the app uses `certifi` to handle certificates automatically. Make sure it's installed:

```bash
pip install --upgrade certifi
```

## Support

For issues or questions, contact the development team.
