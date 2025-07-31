# OCI GenAI Accuracy Assessor

This project provides a comprehensive assessment tool for evaluating Oracle Cloud Infrastructure (OCI) Generative AI models. It measures and scores AI responses across three key dimensions:

- **Factual Accuracy** (60% weight) - How correct and relevant the response is
- **Readability** (20% weight) - How clear and easy to understand the response is  
- **Speed** (20% weight) - How fast the model generates responses

The tool supports multiple OCI GenAI models including Grok 3, Llama 4, Cohere, and Llama 3.3 70B.

## Quick Start

### Prerequisites
- Python 3.x
- OCI Python SDK: `pip install oci`
- Valid OCI configuration and credentials
- Access to OCI Generative AI service

### Basic Usage

1. **Set your compartment ID** (required):
```bash
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1.."
```

2. **Run assessment**:
```bash
python accuracy_checker.py --model grok3 --prompt "What is the capital of France?" --reference "The capital of France is Paris."
```

### Available Models
- `grok3` - Grok 3 
- `llama4` - Llama 4
- `cohere` - Cohere
- `llama33-70b` - Llama 3.3 70B

### Example Output

```
================================================================================
🧠 COMPREHENSIVE OCI GENERATIVE AI ASSESSMENT
================================================================================
 Generating response using Grok 3 (GENERIC) for prompt: What is the capital of France?...
 Response generated successfully
 API Response Time: 0.796s
 Total Tokens: 14
 Tokens/Second: 17.58

 OVERALL SCORE: 4.8/5.0

 COMPONENT SCORES:
    Factual Accuracy: 5.0/5.0 (Weight: 60%)
    Readability: 5.0/5.0 (Weight: 20%)
    Speed: 4.0/5.0 (Weight: 20%)

 WEIGHTED CONTRIBUTIONS:
    Factual Accuracy: 3.0/5.0
    Readability: 1.0/5.0
    Speed: 0.8/5.0

 PERFORMANCE METRICS:
    API Response Time: 0.796s
    Input Tokens: 7
    Output Tokens: 7
    Total Tokens: 14
    Tokens/Second: 17.58

 RESPONSE PREVIEW:
   The capital of France is Paris.

 Results saved to: assessment_results/oci_genai_assessment_20250731_192937.json
```

### Configuration Options

- Use `--help-examples` for detailed usage examples
- Use `--list-models` to see all available models
- Set `OCI_CONFIG_PROFILE` environment variable for custom OCI profiles
- Interactive mode: Run without `--prompt` and `--reference` arguments

### Environment Variables
- `OCI_COMPARTMENT_ID` - Your OCI compartment OCID (required)
- `OCI_CONFIG_PROFILE` - OCI config profile name (default: DEFAULT)
- `OCI_CONFIG_FILE` - Path to OCI config file (default: ~/.oci/config)