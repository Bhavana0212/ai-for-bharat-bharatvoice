<<<<<<< HEAD
#!/usr/bin/env python3
"""Simple import test to check for syntax errors."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService
    print("✅ Successfully imported IndianRailwaysService")
    
    # Test basic instantiation
    service = IndianRailwaysService()
    print("✅ Successfully created service instance")
    
    # Test validation methods
    print(f"✅ Train number validation: {service._validate_train_number('12002')}")
    print(f"✅ PNR validation: {service._validate_pnr('1234567890')}")
    print(f"✅ Date validation: {service._validate_date_format('2024-01-15')}")
    
    print("🎉 All basic tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
=======
#!/usr/bin/env python3
"""Simple import test to check for syntax errors."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService
    print("✅ Successfully imported IndianRailwaysService")
    
    # Test basic instantiation
    service = IndianRailwaysService()
    print("✅ Successfully created service instance")
    
    # Test validation methods
    print(f"✅ Train number validation: {service._validate_train_number('12002')}")
    print(f"✅ PNR validation: {service._validate_pnr('1234567890')}")
    print(f"✅ Date validation: {service._validate_date_format('2024-01-15')}")
    
    print("🎉 All basic tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    sys.exit(1)