"""
Template Validation System for Strategy Deployment
=================================================

Validates deployment templates and ensures they can be successfully deployed
with parameter injection.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import ast
import re

from .deployment_config import DeploymentConfig

logger = logging.getLogger(__name__)


class TemplateValidator:
    """
    Validates deployment templates for correctness and completeness.
    
    Performs static analysis to ensure templates are valid Python code
    and contain required structures for deployment.
    """
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        Initialize template validator.
        
        Args:
            config: Deployment configuration
        """
        self.config = config or DeploymentConfig()
        self.logger = logging.getLogger(__name__)
    
    def validate_template(self, template_path: Path) -> Dict[str, Any]:
        """
        Comprehensive template validation.
        
        Args:
            template_path: Path to template file
            
        Returns:
            Detailed validation result
        """
        result = {
            'template_path': str(template_path),
            'valid': True,
            'errors': [],
            'warnings': [],
            'placeholders': [],
            'required_methods': [],
            'syntax_valid': False,
            'deployment_ready': False
        }
        
        try:
            # Basic file checks
            if not template_path.exists():
                result['valid'] = False
                result['errors'].append(f'Template file not found: {template_path}')
                return result
            
            # Read template content
            try:
                content = template_path.read_text(encoding=self.config.template_encoding)
            except Exception as e:
                result['valid'] = False
                result['errors'].append(f'Failed to read template: {e}')
                return result
            
            if not content.strip():
                result['valid'] = False
                result['errors'].append('Template file is empty')
                return result
            
            # Extract placeholders
            result['placeholders'] = self._extract_placeholders(content)
            
            # Validate Python syntax (with placeholders as dummy values)
            syntax_result = self._validate_syntax(content, result['placeholders'])
            result['syntax_valid'] = syntax_result['valid']
            if not syntax_result['valid']:
                result['valid'] = False
                result['errors'].extend(syntax_result['errors'])
            
            # Validate deployment structure
            structure_result = self._validate_deployment_structure(content)
            result['deployment_ready'] = structure_result['valid']
            result['required_methods'] = structure_result['methods']
            if not structure_result['valid']:
                result['warnings'].extend(structure_result['warnings'])
            
            # Additional checks
            self._validate_parameter_usage(content, result)
            self._check_best_practices(content, result)
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation failed: {e}')
        
        return result
    
    def _extract_placeholders(self, content: str) -> List[str]:
        """Extract all {placeholder} patterns from template (single-line only)."""
        # Only match placeholders that don't contain newlines (to avoid Python dict literals)
        pattern = r'\{([^}\n\r]+)\}'
        matches = re.findall(pattern, content)
        return sorted(list(set(matches)))
    
    def _validate_syntax(self, content: str, placeholders: List[str]) -> Dict[str, Any]:
        """
        Validate Python syntax by substituting placeholders with dummy values.
        
        Args:
            content: Template content
            placeholders: List of placeholder names
            
        Returns:
            Syntax validation result
        """
        try:
            # Create dummy substitutions for placeholders
            test_content = content
            for placeholder in placeholders:
                # Use appropriate dummy values based on placeholder name
                dummy_value = self._get_dummy_value(placeholder)
                test_content = test_content.replace(f'{{{placeholder}}}', str(dummy_value))
            
            # Parse as AST to check syntax
            ast.parse(test_content)
            
            return {'valid': True, 'errors': []}
            
        except SyntaxError as e:
            return {
                'valid': False,
                'errors': [f'Syntax error: {e.msg} at line {e.lineno}']
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Syntax validation failed: {e}']
            }
    
    def _get_dummy_value(self, placeholder_name: str) -> str:
        """Get appropriate dummy value for placeholder based on name patterns."""
        name_lower = placeholder_name.lower()
        
        # String values (without quotes - they're already in the template)
        if any(word in name_lower for word in ['method', 'symbol', 'timeframe', 'name', 'exchange', 'currency', 'start', 'end']):
            return 'dummy_string'
        
        # Boolean values
        if any(word in name_lower for word in ['use_', 'enable_', 'filter', '_flag']):
            return 'True'
        
        # Float values  
        if any(word in name_lower for word in ['multiplier', 'ratio', 'threshold', 'percent', 'std', 'tick_']):
            return '1.0'
        
        # Integer values (periods, counts, sizes)
        if any(word in name_lower for word in ['period', 'bars', 'size', 'count', 'limit']):
            return '10'
        
        # Default to integer
        return '1'
    
    def _validate_deployment_structure(self, content: str) -> Dict[str, Any]:
        """Validate that template has required structure for deployment."""
        warnings = []
        methods = []
        
        # Required components for a deployable strategy
        required_patterns = {
            'class_definition': r'class\s+\w+.*:',
            'generate_signal': r'def\s+generate_signal\s*\(',
            'run_strategy': r'def\s+run_strategy\s*\(',
            'init_method': r'def\s+__init__\s*\('
        }
        
        found_patterns = {}
        for name, pattern in required_patterns.items():
            matches = re.findall(pattern, content)
            found_patterns[name] = len(matches) > 0
            if matches:
                methods.extend(matches)
        
        # Check for required components
        if not found_patterns['class_definition']:
            warnings.append('No class definition found - may not be a valid strategy template')
        
        if not found_patterns['generate_signal'] and 'def process_new_bar' not in content:
            warnings.append('Missing generate_signal or process_new_bar method - required for deployment')
        
        if not found_patterns['run_strategy']:
            warnings.append('Missing run_strategy factory function - recommended for deployment')
        
        if not found_patterns['init_method']:
            warnings.append('Missing __init__ method in strategy class')
        
        # Check fence integrity for simulation blocks
        fence_result = self._validate_fence_integrity(content)
        if not fence_result['valid']:
            warnings.extend(fence_result['issues'])
        
        # Check for common deployment patterns
        if 'position' not in content:
            warnings.append('No position tracking found - may not track strategy state')
        
        if 'bar_data' not in content:
            warnings.append('No bar_data parameter handling found - may not process market data')
        
        deployment_ready = len(warnings) == 0
        
        return {
            'valid': deployment_ready,
            'warnings': warnings,
            'methods': methods,
            'found_patterns': found_patterns
        }
    
    def _validate_parameter_usage(self, content: str, result: Dict[str, Any]):
        """Validate that placeholders are used appropriately."""
        placeholders = result['placeholders']
        
        for placeholder in placeholders:
            # Check if placeholder is used in a reasonable context
            usage_contexts = re.findall(rf'(\w+.*?\{{{placeholder}\}}.*?\w+)', content)
            
            if not usage_contexts:
                result['warnings'].append(
                    f'Placeholder {{{placeholder}}} may not be used in any meaningful context'
                )
    
    def _check_best_practices(self, content: str, result: Dict[str, Any]):
        """Check for deployment best practices."""
        
        # Check for hardcoded values that should be parameters
        hardcoded_patterns = [
            (r'\b\d+\.\d+\b', 'Hardcoded decimal values found - consider parameterizing'),
            (r'\b[1-9]\d{2,}\b', 'Large hardcoded integers found - consider parameterizing')
        ]
        
        for pattern, warning in hardcoded_patterns:
            matches = re.findall(pattern, content)
            if len(matches) > 5:  # Only warn if many hardcoded values
                result['warnings'].append(f'{warning} ({len(matches)} instances)')
        
        # Check for proper error handling
        if 'try:' not in content or 'except' not in content:
            result['warnings'].append('Limited error handling found - consider adding try/except blocks')
        
        # Check for logging
        if 'log' not in content.lower():
            result['warnings'].append('No logging found - consider adding logging for production deployment')
    
    def validate_multiple_templates(self, template_paths: List[Path]) -> Dict[str, Any]:
        """
        Validate multiple templates and provide summary.
        
        Args:
            template_paths: List of template file paths
            
        Returns:
            Combined validation results
        """
        results = {}
        summary = {
            'total_templates': len(template_paths),
            'valid_templates': 0,
            'invalid_templates': 0,
            'templates_with_warnings': 0,
            'common_issues': []
        }
        
        all_issues = []
        
        for template_path in template_paths:
            result = self.validate_template(template_path)
            results[str(template_path)] = result
            
            if result['valid']:
                summary['valid_templates'] += 1
            else:
                summary['invalid_templates'] += 1
            
            if result['warnings']:
                summary['templates_with_warnings'] += 1
            
            # Collect issues for common issue analysis
            all_issues.extend(result['errors'] + result['warnings'])
        
        # Find common issues
        issue_counts = {}
        for issue in all_issues:
            # Generalize issue text for counting
            generalized = re.sub(r'[:\'"]\s*\S+', '', issue)  # Remove specific paths/values
            issue_counts[generalized] = issue_counts.get(generalized, 0) + 1
        
        # Report issues that occur in multiple templates
        summary['common_issues'] = [
            {'issue': issue, 'count': count} 
            for issue, count in issue_counts.items() 
            if count > 1
        ]
        
        return {
            'individual_results': results,
            'summary': summary
        }
    
    def _validate_fence_integrity(self, content: str) -> Dict[str, Any]:
        """
        Validate fence markers for simulation blocks are properly paired and nested.
        
        Args:
            content: Template content to validate
            
        Returns:
            Fence validation result with issues
        """
        issues = []
        
        # Find all fence start and end markers
        start_pattern = r'#\s*FENCE:START:(\w+)'
        end_pattern = r'#\s*FENCE:END:(\w+)'
        
        start_matches = [(m.group(1), m.start()) for m in re.finditer(start_pattern, content)]
        end_matches = [(m.group(1), m.start()) for m in re.finditer(end_pattern, content)]
        
        # Basic count validation
        if len(start_matches) != len(end_matches):
            issues.append(f"Mismatched fence markers: {len(start_matches)} starts, {len(end_matches)} ends")
            return {'valid': False, 'issues': issues}
        
        # Validate each fence block
        fence_stack = []
        all_markers = sorted(
            [(pos, 'start', fence_type) for fence_type, pos in start_matches] + 
            [(pos, 'end', fence_type) for fence_type, pos in end_matches]
        )
        
        for pos, marker_type, fence_type in all_markers:
            if marker_type == 'start':
                fence_stack.append(fence_type)
            else:  # end marker
                if not fence_stack:
                    issues.append(f"Found FENCE:END:{fence_type} without matching FENCE:START")
                    continue
                
                expected_type = fence_stack.pop()
                if expected_type != fence_type:
                    issues.append(f"Fence type mismatch: started with {expected_type}, ended with {fence_type}")
        
        # Check for unclosed fences
        if fence_stack:
            issues.append(f"Unclosed fence blocks: {', '.join(fence_stack)}")
        
        # Validate fence content (basic checks)
        for fence_type, start_pos in start_matches:
            # Find corresponding end marker
            end_pos = None
            for end_type, end_position in end_matches:
                if end_type == fence_type and end_position > start_pos:
                    end_pos = end_position
                    break
            
            if end_pos:
                # Extract fence content
                fence_content = content[start_pos:end_pos]
                
                # Check for nested Python structures that could break stripping
                if fence_content.count('"""') % 2 != 0:
                    issues.append(f"Unmatched triple quotes in FENCE:{fence_type} block")
                
                if fence_content.count("'''") % 2 != 0:
                    issues.append(f"Unmatched single quotes in FENCE:{fence_type} block")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'fence_blocks_found': len(start_matches)
        }