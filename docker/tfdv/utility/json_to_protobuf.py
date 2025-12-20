import json
import os
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_data_validation as tfdv

def convert_json_to_tfdv_schema(json_file_path):
    """
    Reads a custom JSON configuration file and converts it into a TFDV 
    Schema protocol buffer object.
    """
    with open(json_file_path, 'r') as f:
        config = json.load(f)

    # Initialize the TFDV Schema object
    schema = schema_pb2.Schema()
    
    # Map TFDV type strings
    TYPE_MAP = {
        "INT": schema_pb2.FeatureType.INT,
        "FLOAT": schema_pb2.FeatureType.FLOAT,
        "STRING": schema_pb2.FeatureType.STRING
    }

    for feature_config in config.get('features', []):
        feature = schema.feature.add()
        feature.name = feature_config['name']
        feature.type = TYPE_MAP.get(feature_config['type'])

        # --- Set Presence/Value Constraints ---
        
        # Required Check (Presence)
        if feature_config.get('required', False):
            feature.presence.min_fraction = 1.0
        
        # Domain/Allowed Values (Categorical)
        if feature_config.get('allowed_values'):
            domain = feature.domain.add()
            domain.name = feature_config['name']
            domain.value.extend(feature_config['allowed_values'])
            feature.type = schema_pb2.FeatureType.STRING # Ensure type is STRING for domain check
        
        # Value Bounds (Numeric)
        if feature.type in [schema_pb2.FeatureType.INT, schema_pb2.FeatureType.FLOAT]:
            if 'min_value' in feature_config:
                feature.float_domain.min = feature_config['min_value']
            if 'max_value' in feature_config:
                feature.float_domain.max = feature_config['max_value']
        
        # --- Set Drift and Skew Thresholds (Comparative Validation) ---

        # Data Drift Check (L-infinity Norm)
        if 'drift_threshold' in feature_config:
            feature.drift_comparator.infinity_norm.threshold = feature_config['drift_threshold']
        
        # Training/Serving Skew Check (L-infinity Norm)
        if 'skew_threshold' in feature_config:
            feature.skew_comparator.infinity_norm.threshold = feature_config['skew_threshold']

    return schema

# --- Example Execution ---

# 1. Create the dummy JSON file (simulating non-technical team output)
json_config_path = 'feature_config.json'
with open(json_config_path, 'w') as f:
    f.write("""
{
  "project_name": "ecommerce_product",
  "features": [
    {
      "name": "category",
      "type": "STRING",
      "required": true,
      "drift_threshold": 0.05,
      "allowed_values": ["Electronics", "Clothing", "Books", "Home Goods"]
    },
    {
      "name": "price",
      "type": "FLOAT",
      "required": true,
      "min_value": 1.0,
      "max_value": 1000.0
    }
  ]
}
""")

# 2. Run the conversion
tfdv_schema = convert_json_to_tfdv_schema(json_config_path)

# 3. Save the resulting Protobuf text file (.pbtxt)
output_pbtxt_path = 'custom_schema.pbtxt'
tfdv.write_schema_text(tfdv_schema, output_pbtxt_path)

print(f"\nSuccessfully converted {json_config_path} to {output_pbtxt_path}")
print("TFDV Schema generated and ready for use in validation.")

# Clean up
os.remove(json_config_path)
os.remove(output_pbtxt_path)