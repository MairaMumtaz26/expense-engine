"""
QBO Expense Engine - Configuration
Account mappings, expense categories, and keyword rules
"""

# ============================================================
# ACCOUNT MAPPING: QBO Company Name → Brand + Currency
# ============================================================
ACCOUNT_MAP = {
    "2184873 Alberta LTD":    {"brand": "Wholesale",         "currency": "CAD", "type": "Wholesale"},
    "2204162 Alberta LTD":    {"brand": "Wholesale",         "currency": "CAD", "type": "Wholesale"},
    "2204162 Alberta Limited":{"brand": "Wholesale",         "currency": "CAD", "type": "Wholesale"},
    "Fomin LLC":              {"brand": "Fomin LLC",         "currency": "USD", "type": "Private Label"},
    "Kaiya":                  {"brand": "Kaiya",             "currency": "USD", "type": "Private Label"},
    "Khaos Beauty - US":      {"brand": "Wholesale",         "currency": "USD", "type": "Wholesale"},
    "Luna Naturals LLC":      {"brand": "Luna Naturals LLC", "currency": "USD", "type": "Private Label"},
    "Ozuta":                  {"brand": "Wholesale",         "currency": "USD", "type": "Wholesale"},
    "Paper Party LLC":        {"brand": "Paper Party LLC",   "currency": "USD", "type": "Private Label"},
    "Rockport Tools":         {"brand": "Rockport Tools",    "currency": "USD", "type": "Private Label"},
    "Roofus Pet LLC":         {"brand": "Roofus Pet LLC",    "currency": "USD", "type": "Private Label"},
    "Shocca":                 {"brand": "Wholesale",         "currency": "USD", "type": "Wholesale"},
    "Skuxs LLC":              {"brand": "Wholesale",         "currency": "USD", "type": "Wholesale"},
    "WST Holdings Inc":       {"brand": "Wholesale",         "currency": "USD", "type": "Wholesale"},
    "Soul Mama":              {"brand": "Soul Mama",         "currency": "USD", "type": "Private Label"},
}

# Filename keyword fallbacks for matching
FILENAME_KEYWORDS = {
    "2184873": "2184873 Alberta LTD",
    "2204162": "2204162 Alberta LTD",
    "Fomin": "Fomin LLC",
    "Kaiya": "Kaiya",
    "Khaos": "Khaos Beauty - US",
    "Luna": "Luna Naturals LLC",
    "Ozuta": "Ozuta",
    "Paper_Party": "Paper Party LLC",
    "Paper Party": "Paper Party LLC",
    "Rockport": "Rockport Tools",
    "Roofus": "Roofus Pet LLC",
    "Shocca": "Shocca",
    "Skuxs": "Skuxs LLC",
    "WST": "WST Holdings Inc",
    "Soul_Mama": "Soul Mama",
    "Soul Mama": "Soul Mama",
}

# All private label brands (for apportionment targets)
PL_BRANDS = [
    "Fomin LLC", "Kaiya", "Luna Naturals LLC", "Paper Party LLC",
    "Rockport Tools", "Roofus Pet LLC", "Soul Mama"
]

ALL_BRANDS = ["Wholesale"] + PL_BRANDS

# ============================================================
# IGNORED KEYWORDS: Transactions matching these are filtered out
# ============================================================
IGNORED_KEYWORDS = [
    "china office", "pakistan office",
    "china payroll", "pakistan payroll",
    "china", "pakistan",
]

# ============================================================
# EXPENSE CLASSIFICATION: Head → Sub-head keywords
# ============================================================
EXPENSE_CATEGORIES = {
    "Accounting Exp": [
        "QBO", "Gusto", "Patriot"
    ],
    "Bank Charges": [
        "Bank Charges", "Bank Fee", "Service Charge", "Membership Fee",
        "BILL.COM", "Bill.Com"
    ],
    "Financial Charges": [
        "Financial Charges", "Purchase Interest", "Hascap",
        "SellersFI", "Settle", "Lendistry", "Interest", "Clear Co",
        "Financials Charges"
    ],
    "I.T Expense": [
        "I.T Expense", "1Password", "1st Formation", "MarketTime", "Adobe",
        "Affirm", "Sellercloud", "Alura", "Amazon", "Apple", "Archive",
        "AT&T", "Audible", "BambooHR", "Clickup", "Agncy12", "GS1 - UPC",
        "Canva", "Capcut", "Claude.ai", "GoDaddy", "Dropbox", "Freepik",
        "Fineline", "Fireflies", "FormSwift", "Google", "Gorgias", "GS1 UK",
        "Harvest", "Higgsfield", "Hubstaff", "Klaviyo", "LetsGO", "LinkedIn",
        "Microsoft", "Skype", "MirageID", "MoDash", "NameCheap", "Nemoship",
        "OpenAi", "Orderful", "Shopify", "Overjoy", "Pietra", "Figma",
        "GS1", "The Data Council", "Sellerboard", "Shoppayinst", "SKUStack",
        "Slack", "Smarttr", "Spline", "SPS Commerce", "The Rundown AI",
        "QR Code Gen", "TryAtria", "V Platform", "Walmart", "Web EDI",
        "Webflow", "Workable", "M&E", "Podfoods", "Algopix", "BeeHiv",
        "Patriot", "Arlo Technologies", "Wordpress", "Axiom.ai", "Snappering",
        "Sellersnap", "Cal.com", "OpenPhone", "ChatGPT", "CCSI Fax", "Miro",
        "Loom", "Keepa", "Jasper.ai", "Instascraper", "Higgsfield", "Getida",
        "ExpressVPN", "Etsy", "Domain Market", "Datarova", "Datadive",
        "Instascrapper", "Rangeme", "Make.com"
    ],
    "Insurance Exp": [
        "Hartford", "Atradius", "The Guardian", "Nationwide", "Intact",
        "Insurance"
    ],
    "Legal & Prof.": [
        "Matt Goldbloom", "CA Secretary", "Jetstile", "Corp E", "Ecapital",
        "HiTouch", "Agncy12", "LA CPA", "LSQ", "Makers Media",
        "McGovern Law", "ECRM", "Byzzer", "Erewhon", "Donald G",
        "Ranged Ltd", "Ranged", "De Dor", "IRS", "KeHe", "The Founders Club",
        "UL verification", "VAT reg.", "Matt Akins", "M&E", "Nguyen",
        "Filing fee", "Rocket Law", "Tax1099", "DJYJ CPA",
        "Bright Accounting", "FDA Register", "Hampton", "Kalim CPA",
        "Ogden Glazer", "DELAWARE", "DELWARE", "Legal & Professional",
        "American Pet Products"
    ],
    "Marketing & Advert": [
        "Marketing & Advert", "Advertising", "AMAZON.COM*", "Amazon",
        "Fiverr", "Marketing"
    ],
    "Office Exp": [
        "China Office", "Office Exp", "Pakistan Office", "Amazon",
        "Plantation", "M&E", "Amz", "Misc Purchases", "Costco",
        "Plastic Bank", "Misc.", "Shaima", "WeWork", "The ORG",
        "Crown Equipment", "Instacart", "Meal and Entertainment",
        "Meals and entertainment", "Meals and Entertainment",
        "MCDONALD", "DOMINO", "Stater Brothers", "DoorDash",
        "7-Eleven", "GRAB", "Blue Bottle", "HUDSON NEWS"
    ],
    "Rent Exp": [
        "China Office", "Pakistan Office", "US warehouse", "Canada Warehouse",
        "Rent", "Warehouse"
    ],
    "Shipping & Delivery": [
        "Shipping & Delivery", "Shippo", "TQL", "Uline", "Win Pallets",
        "UPS", "ABF*TRANSPORTATION", "City Logistics", "Pallet Company Pro",
        "UniCargo", "DHL", "USPS", "Vendor", "Stamps", "OG Pallets",
        "Slip Enterprises", "Winn Pallets", "FedEx", "Endicia"
    ],
    "Travel Exp": [
        "Travel Exp", "M&E", "WestJet", "Aircanada", "Agoda", "Travel",
        "Hotel", "Flight", "Airline", "Parking"
    ],
    "Utility Exp": [
        "China Office", "Pakistan Office", "Utility Exp", "BayAlarm",
        "Bell Mobility", "Direct Energy", "Edison", "Culligan Water",
        "Epcor", "Frontier", "Garda Alarm", "Primo Water", "Shaw telecom",
        "T Mobile", "SuperSave", "City of Redlands", "BELL MOBILITY",
        "Utilities"
    ],
    "Payroll": [
        "Canada Payroll", "China Payroll", "Pakistan Payroll",
        "Remote Payroll", "USA Payroll", "Payroll", "Gusto", "ADP",
        "Wages", "Gross Pay", "Employer Taxes", "Vacation Pay", "Taxes",
        "Employer Health Insurance"
    ],
    "Consultancy": [
        "Ezat Kaaba", "Zaher Ali Kaaba", "Consultancy"
    ],
}

# Build reverse lookup: keyword → (Head, Sub-head)
def build_keyword_index():
    """Build a keyword lookup for fast matching. Longer keywords matched first."""
    index = []
    for head, keywords in EXPENSE_CATEGORIES.items():
        for kw in keywords:
            index.append((kw.lower(), head, kw))
    # Sort by keyword length descending (longer matches first)
    index.sort(key=lambda x: len(x[0]), reverse=True)
    return index

KEYWORD_INDEX = build_keyword_index()
