import yaml
from typing import List, Dict, Any

class ReasonCodeEngine:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.reason_codes = config.get("reason_codes", [])

    def evaluate(self, attributions: List[Dict[str, Any]]) -> List[str]:
        """
        Evaluates the feature attributions against the predefined rules and returns a list of active reason codes.
        
        Args:
            attributions: List of dicts, each containing:
                          {'feature': str, 'attribution': float, 'direction': str}
                          
        Returns:
            List of reason code strings.
        """
        active_codes = []
        
        # Convert attributions to a lookup for easier evaluation
        attr_lookup = {
            item['feature']: {
                'value': item['attribution'],
                'direction': item.get('direction', 'positive' if item['attribution'] > 0 else 'negative')
            }
            for item in attributions
        }
        
        for rule in self.reason_codes:
            feature_name = rule['feature']
            if feature_name not in attr_lookup:
                continue
                
            attr_info = attr_lookup[feature_name]
            
            # Check direction
            expected_direction = rule.get('direction', 'positive')
            
            # Rule: only evaluate if attribution is in the expected direction
            # If expected_direction is positive, we want attribution > 0.
            if expected_direction == 'positive' and attr_info['value'] <= 0:
                continue
            if expected_direction == 'negative' and attr_info['value'] >= 0:
                continue
                
            # Check minimum attribution magnitude threshold
            min_attr = rule.get('min_attribution', 0.0)
            if abs(attr_info['value']) >= min_attr:
                active_codes.append(rule['code'])
                
        return active_codes
