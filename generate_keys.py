"""
License Key Generator for DatadoneV3
Usage: python generate_keys.py
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from DatadoneV3 import SessionLocal, LicenseKey, datetime, timedelta, secrets

def generate_key(tier: str, company_id: str) -> str:
    """Generate a single license key"""
    db = SessionLocal()
    try:
        # Generate key
        prefix = f"DD3-{tier.upper()}-{datetime.now().year}"
        suffix = secrets.token_hex(6).upper()
        key = f"{prefix}-{suffix}"
        
        # Store in database
        license = LicenseKey(
            key=key,
            company_id=company_id,
            tier=tier,
            max_analyses=get_limit(tier),
            analyses_used=0,
            expiry_date=datetime.utcnow() + timedelta(days=365),
            is_active=1
        )
        db.add(license)
        db.commit()
        
        return key
    finally:
        db.close()

def get_limit(tier: str) -> int:
    """Get analysis limit for tier"""
    limits = {
        "starter": 100,
        "professional": 1000,
        "enterprise": 10000
    }
    return limits.get(tier, 1000)

def generate_multiple_keys(tier: str, companies: list):
    """Generate keys for multiple companies"""
    print(f"\n{'='*60}")
    print(f"Generating {len(companies)} {tier.upper()} license keys...")
    print(f"{'='*60}\n")
    
    results = []
    for company in companies:
        key = generate_key(tier, company)
        results.append((company, key))
        print(f"Company: {company}")
        print(f"Key:     {key}")
        print(f"TIER:    {tier.upper()} | LIMIT: {get_limit(tier)} analyses")
        print(f"EXPIRY:  1 year from today")
        print("-" * 60)
    
    # Also save to CSV
    save_to_csv(results, tier)
    print(f"\n✅ Keys saved to license_keys_{tier}.csv")
    return results

def save_to_csv(results: list, tier: str):
    """Save keys to CSV file"""
    import csv
    with open(f"license_keys_{tier}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Company", "License Key", "Tier", "Limit", "Expiry Date"])
        for company, key in results:
            writer.writerow([
                company,
                key,
                tier,
                get_limit(tier),
                (datetime.utcnow() + timedelta(days=365)).strftime("%Y-%m-%d")
            ])

def interactive_mode():
    """Interactive menu"""
    print("\n" + "="*60)
    print(" DATADONEV3 LICENSE KEY GENERATOR")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Generate a single key")
        print("2. Generate multiple keys from list")
        print("3. Generate keys from file")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            tier = input("Enter tier (starter/professional/enterprise): ").lower()
            company = input("Enter company name: ")
            key = generate_key(tier, company)
            print(f"\n✅ Key generated: {key}")
            
        elif choice == "2":
            tier = input("Enter tier (starter/professional/enterprise): ").lower()
            companies = []
            print("Enter company names (one per line, empty line to finish):")
            while True:
                company = input("> ")
                if company.strip() == "":
                    break
                companies.append(company)
            generate_multiple_keys(tier, companies)
            
        elif choice == "3":
            filename = input("Enter filename with company names (one per line): ")
            try:
                with open(filename, 'r') as f:
                    companies = [line.strip() for line in f if line.strip()]
                tier = input("Enter tier (starter/professional/enterprise): ").lower()
                generate_multiple_keys(tier, companies)
            except FileNotFoundError:
                print(f"❌ File {filename} not found")
        
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("❌ Invalid choice")

def quick_start():
    """Generate 5 sample keys immediately"""
    sample_companies = [
        "TechCorp Ltd",
        "LocalShop Inc",
        "StartupXYZ",
        "EcommerceStore",
        "ConsultingFirm"
    ]
    
    print("\n" + "="*60)
    print(" QUICK START: Generating 5 Professional Keys")
    print("="*60)
    
    keys = generate_multiple_keys("professional", sample_companies)
    
    print("\n🎉 Done! Here are your keys:")
    for company, key in keys:
        print(f"{company}: {key}")

if __name__ == "__main__":
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_start()
        else:
            print("Usage: python generate_keys.py [--quick]")
            print("   --quick: Generate 5 sample professional keys immediately")
    else:
        interactive_mode()