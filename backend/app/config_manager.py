"""
Configuration manager for dynamic disease detection system.
Handles loading and validation of disease configuration data.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreatmentInfo:
    """Treatment information for a disease."""
    chemical: str
    cultural: str
    preventive: str


@dataclass
class DiseaseClass:
    """Disease class configuration."""
    id: str
    name: str
    plant: str
    disease: str
    severity: str
    treatment: TreatmentInfo
    symptoms: Optional[List[str]] = None
    causes: Optional[str] = None
    urgency: Optional[str] = None
    economic_impact: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    version: str
    description: str
    framework: str
    input_shape: List[int]
    supported_formats: List[str]


@dataclass
class SeverityLevel:
    """Severity level configuration."""
    level: int
    color: str
    description: str


@dataclass
class TreatmentType:
    """Treatment type configuration."""
    name: str
    description: str


@dataclass
class ConfidenceThresholds:
    """Confidence threshold configuration."""
    high_confidence: float
    medium_confidence: float
    low_confidence: float
    uncertain: float


class ConfigManager:
    """Manages disease detection configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Default to config directory relative to this file
            config_dir = Path(__file__).parent
            config_path = config_dir / "disease_config.json"
        
        self.config_path = Path(config_path)
        self._config = None
        self._disease_classes = None
        self._model_config = None
        self._severity_levels = None
        self._treatment_types = None
        self._confidence_thresholds = None
        
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.json':
                    self._config = json.load(f)
                elif self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    self._config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {self.config_path.suffix}")
            
            self._parse_config()
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _parse_config(self) -> None:
        """Parse loaded configuration into dataclasses."""
        # Parse model config
        model_data = self._config.get('model', {})
        self._model_config = ModelConfig(**model_data)
        
        # Parse disease classes
        self._disease_classes = {}
        for class_data in self._config.get('classes', []):
            treatment_data = class_data.get('treatment', {})
            treatment = TreatmentInfo(**treatment_data)
            
            disease_class = DiseaseClass(
                id=class_data['id'],
                name=class_data['name'],
                plant=class_data['plant'],
                disease=class_data['disease'],
                severity=class_data['severity'],
                treatment=treatment,
                symptoms=class_data.get('symptoms', []),
                causes=class_data.get('causes', ''),
                urgency=class_data.get('urgency', ''),
                economic_impact=class_data.get('economic_impact', '')
            )
            self._disease_classes[disease_class.id] = disease_class
        
        # Parse severity levels
        self._severity_levels = {}
        for level_name, level_data in self._config.get('severity_levels', {}).items():
            self._severity_levels[level_name] = SeverityLevel(**level_data)
        
        # Parse treatment types
        self._treatment_types = {}
        for type_name, type_data in self._config.get('treatment_types', {}).items():
            self._treatment_types[type_name] = TreatmentType(**type_data)
        
        # Parse confidence thresholds
        threshold_data = self._config.get('confidence_thresholds', {})
        self._confidence_thresholds = ConfidenceThresholds(**threshold_data)
    
    @property
    def model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self._model_config
    
    @property
    def disease_classes(self) -> Dict[str, DiseaseClass]:
        """Get disease classes configuration."""
        return self._disease_classes
    
    @property
    def severity_levels(self) -> Dict[str, SeverityLevel]:
        """Get severity levels configuration."""
        return self._severity_levels
    
    @property
    def treatment_types(self) -> Dict[str, TreatmentType]:
        """Get treatment types configuration."""
        return self._treatment_types
    
    @property
    def confidence_thresholds(self) -> ConfidenceThresholds:
        """Get confidence thresholds configuration."""
        return self._confidence_thresholds
    
    def get_class_names(self) -> List[str]:
        """Get list of all class IDs."""
        return list(self._disease_classes.keys())
    
    def get_class_by_id(self, class_id: str) -> Optional[DiseaseClass]:
        """Get disease class by ID."""
        return self._disease_classes.get(class_id)
    
    def get_treatment_for_class(self, class_id: str) -> Optional[TreatmentInfo]:
        """Get treatment information for a disease class."""
        disease_class = self.get_class_by_id(class_id)
        return disease_class.treatment if disease_class else None
    
    def get_severity_for_class(self, class_id: str) -> Optional[SeverityLevel]:
        """Get severity level for a disease class."""
        disease_class = self.get_class_by_id(class_id)
        if disease_class:
            return self._severity_levels.get(disease_class.severity)
        return None
    
    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level name based on confidence score."""
        thresholds = self._confidence_thresholds
        
        if confidence >= thresholds.high_confidence:
            return "high"
        elif confidence >= thresholds.medium_confidence:
            return "medium"
        elif confidence >= thresholds.low_confidence:
            return "low"
        else:
            return "uncertain"
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate model config
        if not self._model_config:
            errors.append("Model configuration is missing")
        elif not all([self._model_config.name, self._model_config.framework]):
            errors.append("Model name and framework are required")
        
        # Validate disease classes
        if not self._disease_classes:
            errors.append("No disease classes defined")
        
        for class_id, disease_class in self._disease_classes.items():
            if not disease_class.treatment:
                errors.append(f"Treatment information missing for class: {class_id}")
            
            if disease_class.severity not in self._severity_levels:
                errors.append(f"Invalid severity level '{disease_class.severity}' for class: {class_id}")
        
        # Validate confidence thresholds
        if self._confidence_thresholds:
            thresholds = self._confidence_thresholds
            if not (0 <= thresholds.uncertain <= thresholds.low_confidence <= 
                   thresholds.medium_confidence <= thresholds.high_confidence <= 1):
                errors.append("Confidence thresholds must be in ascending order between 0 and 1")
        
        return errors
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.load_config()
    
    def export_legacy_labels(self) -> List[str]:
        """Export class IDs as legacy labels list for compatibility."""
        return self.get_class_names()
    
    def export_legacy_treatments(self) -> Dict[str, str]:
        """Export treatments as legacy dictionary for compatibility."""
        treatments = {}
        for class_id, disease_class in self._disease_classes.items():
            # Combine all treatment types into a single string
            treatment_parts = []
            if disease_class.treatment.chemical:
                treatment_parts.append(disease_class.treatment.chemical)
            if disease_class.treatment.cultural:
                treatment_parts.append(disease_class.treatment.cultural)
            treatments[class_id] = ". ".join(treatment_parts)
        return treatments


# Global configuration manager instance
config_manager = ConfigManager()