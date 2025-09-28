"""
Expand disease configuration with additional common plant diseases.
"""

import json

def expand_disease_config():
    """Add more plant diseases to the configuration."""
    
    # Load current configuration
    config_path = "../backend/app/disease_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Additional diseases to add
    additional_diseases = [
        {
            "id": "Mango___Anthracnose",
            "name": "Mango Anthracnose",
            "plant": "Mango",
            "disease": "Anthracnose",
            "severity": "high",
            "symptoms": [
                "Dark sunken lesions on fruit",
                "Black spots with pink spore masses",
                "Fruit rot starting from blossom end",
                "Leaf spots with brown centers",
                "Premature fruit drop"
            ],
            "causes": "Caused by fungus Colletotrichum gloeosporioides, favored by warm humid conditions",
            "treatment": {
                "chemical": "Apply copper-based fungicides or azoxystrobin during flowering and fruit development. Repeat every 10-14 days.",
                "cultural": "Remove infected fruit and debris. Improve air circulation through proper pruning. Harvest fruit at proper maturity.",
                "preventive": "Plant resistant varieties. Ensure proper tree spacing. Avoid overhead irrigation during flowering."
            },
            "urgency": "High - treat immediately to prevent fruit loss",
            "economic_impact": "Can cause 50-80% fruit loss in humid conditions"
        },
        {
            "id": "Mango___Healthy",
            "name": "Healthy Mango",
            "plant": "Mango",
            "disease": "None",
            "severity": "none",
            "symptoms": [
                "Dark green, glossy leaves",
                "No visible spots or discoloration",
                "Healthy fruit development",
                "Normal flowering pattern",
                "No premature fruit drop"
            ],
            "causes": "No disease present - optimal tropical growing conditions maintained",
            "treatment": {
                "chemical": "No treatment needed - maintain preventive spray schedule during monsoon",
                "cultural": "Continue regular care including proper watering, fertilization, and pruning. Monitor for pest and disease pressure.",
                "preventive": "Maintain good orchard hygiene. Apply preventive copper sprays before monsoon. Ensure proper drainage."
            },
            "urgency": "None - maintain current practices",
            "economic_impact": "Optimal fruit yield and quality expected"
        },
        {
            "id": "Citrus___Canker",
            "name": "Citrus Canker",
            "plant": "Citrus",
            "disease": "Citrus Canker",
            "severity": "very_high",
            "symptoms": [
                "Raised corky lesions on leaves",
                "Water-soaked spots with yellow halos",
                "Circular brown lesions on fruit",
                "Premature fruit and leaf drop",
                "Twig dieback in severe cases"
            ],
            "causes": "Caused by bacteria Xanthomonas axonopodis, spread by wind and rain",
            "treatment": {
                "chemical": "Apply copper-based bactericides preventively. Use streptomycin in early stages (where permitted).",
                "cultural": "Remove and destroy infected plant material. Disinfect tools between plants. Improve air circulation.",
                "preventive": "Plant resistant varieties. Avoid overhead irrigation. Implement quarantine measures for new plants."
            },
            "urgency": "Critical - immediate treatment and isolation required",
            "economic_impact": "Can cause complete crop loss and tree mortality"
        },
        {
            "id": "Rice___Blast",
            "name": "Rice Blast",
            "plant": "Rice",
            "disease": "Rice Blast",
            "severity": "high",
            "symptoms": [
                "Diamond-shaped lesions on leaves",
                "Gray centers with brown margins",
                "Neck rot causing panicle death",
                "Node infection causing lodging",
                "Reduced grain filling"
            ],
            "causes": "Caused by fungus Magnaporthe oryzae, favored by high humidity and moderate temperatures",
            "treatment": {
                "chemical": "Apply systemic fungicides like tricyclazole or azoxystrobin at early disease stages.",
                "cultural": "Use balanced fertilization avoiding excess nitrogen. Ensure proper field drainage. Remove crop residues.",
                "preventive": "Plant resistant varieties. Use certified disease-free seeds. Practice crop rotation."
            },
            "urgency": "High - treat within 3-5 days to prevent epidemic",
            "economic_impact": "Can reduce yield by 30-70% in susceptible varieties"
        },
        {
            "id": "Wheat___Rust",
            "name": "Wheat Rust",
            "plant": "Wheat",
            "disease": "Stripe Rust",
            "severity": "high",
            "symptoms": [
                "Yellow stripes on leaf surfaces",
                "Rusty orange spores on leaves",
                "Premature leaf senescence",
                "Reduced grain size and weight",
                "Weakened plant structure"
            ],
            "causes": "Caused by fungus Puccinia striiformis, spread by airborne spores",
            "treatment": {
                "chemical": "Apply triazole or strobilurin fungicides at first sign of disease. Repeat applications as needed.",
                "cultural": "Remove volunteer wheat plants. Plant at recommended seeding rates. Ensure adequate nutrition.",
                "preventive": "Use resistant wheat varieties. Monitor weather conditions. Plant at optimal times."
            },
            "urgency": "High - treat immediately to prevent spread",
            "economic_impact": "Can reduce yield by 20-40% if left untreated"
        }
    ]
    
    # Add new diseases to existing classes
    config["classes"].extend(additional_diseases)
    
    # Update the configuration file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Added {len(additional_diseases)} new diseases to configuration")
    print(f"Total diseases now: {len(config['classes'])}")
    
    # Also update the labels.json file
    labels_path = "../backend/labels.json"
    labels = [disease["id"] for disease in config["classes"]]
    
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Updated labels.json with {len(labels)} classes")

if __name__ == "__main__":
    expand_disease_config()