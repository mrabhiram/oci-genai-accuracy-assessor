# coding: utf-8
# Copyright (c) 2023, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

##########################################################################
# accuracy_checker.py
# Supports Python 3
##########################################################################
# Info:
# Get texts from LLM model and assess them for factual accuracy, readability, and speed
# using OCI Generative AI Service with comprehensive metrics collection.
#
# Usage Examples:
# 1. Using command line arguments:
#    python accuracy_checker.py --compartment-id "ocid1.compartment..." --model grok3 \
#           --prompt "What is AI?" --reference "AI is artificial intelligence"
#
# 2. Using environment variables:
#    export OCI_COMPARTMENT_ID="ocid1.compartment..."
#    export OCI_CONFIG_PROFILE="MY_PROFILE"
#    python accuracy_checker.py --model grok3 --prompt "What is AI?" --reference "AI is artificial intelligence"
#
# 3. Interactive mode:
#    python accuracy_checker.py --compartment-id "ocid1.compartment..." --model grok3
#
# Required Configuration:
# - Compartment ID (via --compartment-id or OCI_COMPARTMENT_ID env var)
# - Valid OCI config file (default: ~/.oci/config)
# - OCI credentials with GenAI service access
##########################################################################

import oci
import time
import re
import json
import argparse
import os
from datetime import datetime
from difflib import SequenceMatcher

class OCIGenAIAssessor:
    def __init__(self, compartment_id, config_profile="DEFAULT", config_file=None):
        """Initialize OCI Generative AI client and assessment tools"""
        
        # Validate required parameters
        if not compartment_id:
            raise ValueError("Compartment ID is required")
        
        self.compartment_id = compartment_id
        
        # Load OCI config from specified file or default location
        if config_file:
            self.config = oci.config.from_file(config_file, config_profile)
        else:
            self.config = oci.config.from_file('~/.oci/config', config_profile)
        
        # Set endpoint - default to Chicago region
        endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        
        # Initialize OCI client with timeout settings
        self.generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=self.config, 
            service_endpoint=endpoint, 
            retry_strategy=oci.retry.NoneRetryStrategy(), 
            timeout=(10, 240)
        )
        
        # Model configurations with their specific settings
        self.model_configs = {
            'llama4': {
                'model_id': 'ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyayjawvuonfkw2ua4bob4rlnnlhs522pafbglivtwlfzta',
                'api_format': 'GENERIC',
                'max_tokens': 4096,
                'top_k': -1,
                'name': 'Llama 4'
            },
            'cohere': {
                'model_id': 'ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyapnibwg42qjhwaxrlqfpreueirtwghiwvv2whsnwmnlva',
                'api_format': 'COHERE', 
                'max_tokens': 4000,
                'top_k': 0,
                'name': 'Cohere'
            },
            'grok3': {
                'model_id': 'ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya6dvgvvj3ovy4lerdl6fvx525x3yweacnrgn4ryfwwcoq',
                'api_format': 'GENERIC',
                'max_tokens': 4000,
                'top_k': -1,
                'name': 'Grok 3'
            },
            'llama33-70b': {
                'model_id': 'ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyajqi26fkxly6qje5ysvezzrypapl7ujdnqfjq6hzo2loq',
                'api_format': 'GENERIC',
                'max_tokens': 4096,
                'top_k': -1,
                'name': 'Llama 3.3 70B'
            }
        }
        
        # Common stop words for assessment
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'with', 'have', 'had', 'will', 'would', 'could', 'should',
            'this', 'these', 'they', 'them', 'their', 'there', 'where', 'when',
            'what', 'which', 'who', 'why', 'how', 'or', 'but', 'not', 'can'
        }
        
        print(" OCI Generative AI Assessor initialized successfully")
    
    def parse_response_text(self, response_data, api_format):
        """Parse response text based on API format"""
        try:
            if api_format == 'COHERE':
                # COHERE format response parsing
                if hasattr(response_data, 'chat_response') and response_data.chat_response:
                    if hasattr(response_data.chat_response, 'text'):
                        return response_data.chat_response.text
                    elif hasattr(response_data.chat_response, 'chat_history'):
                        # Get the last bot response from chat history
                        chat_history = response_data.chat_response.chat_history
                        for entry in reversed(chat_history):
                            if hasattr(entry, 'role') and entry.role == 'CHATBOT':
                                return getattr(entry, 'message', str(entry))
                
            else:  # GENERIC format
                # GENERIC format response parsing
                if hasattr(response_data, 'chat_response') and response_data.chat_response:
                    if hasattr(response_data.chat_response, 'choices') and response_data.chat_response.choices:
                        choice = response_data.chat_response.choices[0]
                        if hasattr(choice, 'message') and choice.message:
                            if hasattr(choice.message, 'content') and choice.message.content:
                                return choice.message.content[0].text
            
            # Fallback to string representation
            return str(response_data)
            
        except Exception as e:
            print(f" Warning: Error parsing response text: {e}")
            return str(response_data)
    
    def get_model_config(self, model_key):
        """Get model configuration by key or model ID"""
        # If it's a direct model key
        if model_key in self.model_configs:
            return self.model_configs[model_key]
        
        # If it's a direct model ID, try to find it
        for key, config in self.model_configs.items():
            if config['model_id'] == model_key:
                return config
        
        # Default to generic configuration for unknown models
        return {
            'model_id': model_key,
            'api_format': 'GENERIC',
            'max_tokens': 4000,
            'top_k': -1,
            'name': 'Unknown Model'
        }
    
    def generate_response_with_metrics(self, prompt: str, model_key: str, 
                                     max_tokens=None, temperature=0.7) -> dict:
        """Generate response from OCI GenAI and capture comprehensive metrics"""
        
        # Get model configuration
        model_config = self.get_model_config(model_key)
        model_id = model_config['model_id']
        api_format = model_config['api_format']
        model_max_tokens = model_config['max_tokens']
        top_k_value = model_config['top_k']
        model_name = model_config['name']
        
        # Use model's max tokens if not specified
        if max_tokens is None or max_tokens > model_max_tokens:
            max_tokens = model_max_tokens
        
        print(f" Generating response using {model_name} ({api_format}) for prompt: {prompt[:50]}...")
        
        # Record start time for total latency
        start_time = time.time()
        
        try:
            # Setup chat details
            chat_detail = oci.generative_ai_inference.models.ChatDetails()
            
            if api_format == 'COHERE':
                # Setup COHERE chat request
                chat_request = oci.generative_ai_inference.models.CohereChatRequest()
                chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_COHERE
                chat_request.message = prompt
                chat_request.max_tokens = max_tokens
                chat_request.temperature = temperature
                chat_request.frequency_penalty = 0
                chat_request.presence_penalty = 0
                chat_request.top_p = 1
                chat_request.top_k = top_k_value
                
            else:  # GENERIC format
                # Setup GENERIC chat request with message structure
                content = oci.generative_ai_inference.models.TextContent()
                content.text = prompt
                message = oci.generative_ai_inference.models.Message()
                message.role = "USER"
                message.content = [content]
                
                chat_request = oci.generative_ai_inference.models.GenericChatRequest()
                chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
                chat_request.messages = [message]
                chat_request.max_tokens = max_tokens
                chat_request.temperature = temperature
                chat_request.frequency_penalty = 0
                chat_request.presence_penalty = 0
                chat_request.top_p = 1
                chat_request.top_k = top_k_value
            
            # Setup serving mode
            chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id)
            chat_detail.chat_request = chat_request
            chat_detail.compartment_id = self.compartment_id
            
            # Record API call start time
            api_start_time = time.time()
            
            # Make the API call
            chat_response = self.generative_ai_inference_client.chat(chat_detail)
            
            # Record end times
            api_end_time = time.time()
            total_end_time = time.time()
            
            # Calculate timing metrics
            api_response_time = api_end_time - api_start_time
            total_response_time = total_end_time - start_time
            
            # Extract response data
            response_data = chat_response.data
            
            # Parse response based on API format
            response_text = self.parse_response_text(response_data, api_format)
            
            # Extract token usage information
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            
            if hasattr(response_data, 'chat_response') and response_data.chat_response:
                if hasattr(response_data.chat_response, 'usage') and response_data.chat_response.usage:
                    usage = response_data.chat_response.usage
                    input_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0) or (input_tokens + output_tokens)
            
            # If no token info available, estimate
            if total_tokens == 0:
                input_tokens = self.estimate_token_count(prompt)
                output_tokens = self.estimate_token_count(response_text)
                total_tokens = input_tokens + output_tokens
            
            # Calculate tokens per second
            tokens_per_second = total_tokens / api_response_time if api_response_time > 0 else 0
            
            return {
                'response_text': response_text,
                'success': True,
                'metrics': {
                    'api_response_time': round(api_response_time, 3),
                    'total_response_time': round(total_response_time, 3),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'tokens_per_second': round(tokens_per_second, 2),
                    'model_id': model_id,
                    'timestamp': datetime.now().isoformat()
                },
                'raw_response': response_data
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f" Error generating response: {str(e)}")
            return {
                'response_text': f"Error: {str(e)}",
                'success': False,
                'metrics': {
                    'api_response_time': error_time,
                    'total_response_time': error_time,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'tokens_per_second': 0,
                    'model_id': model_id,
                    'timestamp': datetime.now().isoformat()
                },
                'error': str(e)
            }
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count (roughly 4 characters per token for English)"""
        return max(1, len(text) // 4)
    
    def assess_factual_accuracy(self, response: str, reference: str) -> float:
        """Assess factual accuracy (0-5 scale) - 60% weight"""
        
        if not response.strip() or not reference.strip():
            return 0.0
        
        # 1. Text similarity
        similarity_score = self.calculate_text_similarity(response, reference)
        
        # 2. Key facts overlap
        facts_score = self.calculate_factual_overlap(response, reference)
        
        # 3. Keyword coverage
        keyword_score = self.calculate_keyword_coverage(response, reference)
        
        # 4. Numerical accuracy
        numerical_score = self.calculate_numerical_accuracy(response, reference)
        
        # Combine factual accuracy components
        factual_accuracy = (
            similarity_score * 0.4 +      # Semantic similarity
            facts_score * 0.3 +           # Factual elements overlap
            keyword_score * 0.2 +         # Keyword coverage
            numerical_score * 0.1         # Numerical accuracy
        )
        
        return min(5.0, factual_accuracy * 5)
    
    def assess_readability(self, response: str) -> float:
        """Assess readability (0-5 scale) - 20% weight"""
        
        if not response.strip():
            return 0.0
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        words = response.split()
        
        if not sentences or not words:
            return 1.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / len(words)
        
        # Sentence length scoring
        if avg_sentence_length <= 15 and avg_word_length <= 5:
            sentence_score = 5.0
        elif avg_sentence_length <= 20 and avg_word_length <= 6:
            sentence_score = 4.0
        elif avg_sentence_length <= 25 and avg_word_length <= 7:
            sentence_score = 3.0
        elif avg_sentence_length <= 30 and avg_word_length <= 8:
            sentence_score = 2.0
        else:
            sentence_score = 1.0
        
        # Complexity scoring
        complex_words = sum(1 for word in words if len(word.strip('.,!?;:')) > 7)
        complexity_ratio = complex_words / len(words) if words else 0
        
        if complexity_ratio <= 0.1:
            complexity_score = 5.0
        elif complexity_ratio <= 0.2:
            complexity_score = 4.0
        elif complexity_ratio <= 0.3:
            complexity_score = 3.0
        elif complexity_ratio <= 0.4:
            complexity_score = 2.0
        else:
            complexity_score = 1.0
        
        readability_score = (sentence_score * 0.6 + complexity_score * 0.4)
        return min(5.0, readability_score)
    
    def assess_speed(self, metrics: dict) -> float:
        """Assess response speed (0-5 scale) - 20% weight"""
        
        response_time = metrics.get('api_response_time', 0)
        tokens_per_second = metrics.get('tokens_per_second', 0)
        
        speed_scores = []
        
        # Response time scoring
        if response_time <= 1.0:
            time_score = 5.0
        elif response_time <= 2.0:
            time_score = 4.5
        elif response_time <= 3.0:
            time_score = 4.0
        elif response_time <= 5.0:
            time_score = 3.5
        elif response_time <= 8.0:
            time_score = 3.0
        elif response_time <= 12.0:
            time_score = 2.5
        elif response_time <= 20.0:
            time_score = 2.0
        elif response_time <= 30.0:
            time_score = 1.5
        else:
            time_score = 1.0
        
        speed_scores.append(time_score)
        
        # Tokens per second scoring
        if tokens_per_second >= 100:
            tps_score = 5.0
        elif tokens_per_second >= 50:
            tps_score = 4.5
        elif tokens_per_second >= 30:
            tps_score = 4.0
        elif tokens_per_second >= 20:
            tps_score = 3.5
        elif tokens_per_second >= 15:
            tps_score = 3.0
        elif tokens_per_second >= 10:
            tps_score = 2.5
        elif tokens_per_second >= 5:
            tps_score = 2.0
        elif tokens_per_second >= 2:
            tps_score = 1.5
        else:
            tps_score = 1.0
        
        speed_scores.append(tps_score)
        
        return sum(speed_scores) / len(speed_scores)
    
    def calculate_text_similarity(self, response: str, reference: str) -> float:
        """Calculate text similarity using sequence matching"""
        return SequenceMatcher(None, response.lower(), reference.lower()).ratio()
    
    def calculate_factual_overlap(self, response: str, reference: str) -> float:
        """Calculate overlap of factual elements"""
        response_facts = self.extract_facts(response)
        reference_facts = self.extract_facts(reference)
        
        if not reference_facts:
            return 1.0
        
        overlap = len(response_facts.intersection(reference_facts))
        return overlap / len(reference_facts)
    
    def calculate_keyword_coverage(self, response: str, reference: str) -> float:
        """Calculate keyword coverage"""
        def extract_keywords(text):
            words = re.findall(r'\b\w+\b', text.lower())
            keywords = [word for word in words if word not in self.stopwords and len(word) > 2]
            return set(keywords)
        
        response_keywords = extract_keywords(response)
        reference_keywords = extract_keywords(reference)
        
        if not reference_keywords:
            return 1.0
        
        overlap = len(response_keywords.intersection(reference_keywords))
        return overlap / len(reference_keywords)
    
    def calculate_numerical_accuracy(self, response: str, reference: str) -> float:
        """Calculate numerical accuracy"""
        response_numbers = set(re.findall(r'\$?[\d,]+\.?\d*%?', response))
        reference_numbers = set(re.findall(r'\$?[\d,]+\.?\d*%?', reference))
        
        if not reference_numbers:
            return 1.0
        
        correct_numbers = len(response_numbers.intersection(reference_numbers))
        return correct_numbers / len(reference_numbers)
    
    def extract_facts(self, text: str) -> set:
        """Extract factual elements from text"""
        facts = set()
        
        # Numbers and percentages
        numbers = re.findall(r'\$?[\d,]+\.?\d*%?', text)
        facts.update(numbers)
        
        # Years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        facts.update(years)
        
        # Proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        facts.update(proper_nouns)
        
        return facts
    
    def comprehensive_assessment(self, prompt: str, reference_answer: str, 
                                model_key: str, max_tokens=None, temperature=0.7) -> dict:
        """Perform comprehensive assessment with OCI GenAI"""
        
        print("\n" + "="*80)
        print("🧠 COMPREHENSIVE OCI GENERATIVE AI ASSESSMENT")
        print("="*80)
        
        # Generate response with metrics
        result = self.generate_response_with_metrics(
            prompt, model_key, max_tokens, temperature
        )
        
        if not result['success']:
            return {
                'overall_score': 0.0,
                'error': result.get('error', 'Unknown error'),
                'metrics': result['metrics']
            }
        
        response_text = result['response_text']
        metrics = result['metrics']
        
        print(f" Response generated successfully")
        print(f" API Response Time: {metrics['api_response_time']}s")
        print(f" Total Tokens: {metrics['total_tokens']}")
        print(f" Tokens/Second: {metrics['tokens_per_second']}")
        
        # Perform assessments
        factual_accuracy = self.assess_factual_accuracy(response_text, reference_answer)
        readability = self.assess_readability(response_text)
        speed = self.assess_speed(metrics)
        
        # Calculate weighted score: Factual (60%), Readability (20%), Speed (20%)
        weighted_score = (
            factual_accuracy * 0.60 +
            readability * 0.20 +
            speed * 0.20
        )
        
        assessment_result = {
            'overall_score': round(weighted_score, 2),
            'components': {
                'factual_accuracy': round(factual_accuracy, 2),
                'readability': round(readability, 2),
                'speed': round(speed, 2)
            },
            'weighted_contributions': {
                'factual_accuracy': round(factual_accuracy * 0.60, 2),
                'readability': round(readability * 0.20, 2),
                'speed': round(speed * 0.20, 2)
            },
            'response_text': response_text,
            'performance_metrics': metrics,
            'success': True
        }
        
        self.print_assessment_results(assessment_result, prompt, reference_answer)
        
        return assessment_result
    
    def print_assessment_results(self, results: dict, prompt: str, reference: str):
        """Print formatted assessment results"""
        
        print(f"\n OVERALL SCORE: {results['overall_score']}/5.0")
        
        
        print(f"\n COMPONENT SCORES:")
        print(f"    Factual Accuracy: {results['components']['factual_accuracy']}/5.0 (Weight: 60%)")
        print(f"    Readability: {results['components']['readability']}/5.0 (Weight: 20%)")
        print(f"    Speed: {results['components']['speed']}/5.0 (Weight: 20%)")
        
        print(f"\n WEIGHTED CONTRIBUTIONS:")
        print(f"    Factual Accuracy: {results['weighted_contributions']['factual_accuracy']}/5.0")
        print(f"    Readability: {results['weighted_contributions']['readability']}/5.0")
        print(f"    Speed: {results['weighted_contributions']['speed']}/5.0")
        
        print(f"\n PERFORMANCE METRICS:")
        metrics = results['performance_metrics']
        print(f"    API Response Time: {metrics['api_response_time']}s")
        print(f"    Input Tokens: {metrics['input_tokens']}")
        print(f"    Output Tokens: {metrics['output_tokens']}")
        print(f"    Total Tokens: {metrics['total_tokens']}")
        print(f"    Tokens/Second: {metrics['tokens_per_second']}")
        print(f"    Model ID: {metrics['model_id']}")
        
        print(f"\n RESPONSE PREVIEW:")
        response_preview = results['response_text'][:300] + "..." if len(results['response_text']) > 300 else results['response_text']
        print(f"   {response_preview}")

def print_usage_examples():
    """Print usage examples and configuration help"""
    print("\n" + "="*80)
    print("📖 USAGE EXAMPLES")
    print("="*80)
    print("\n1. Command line arguments:")
    print('   python accuracy_checker.py --compartment-id "ocid1.compartment..." \\')
    print('          --model grok3 --prompt "What is AI?" --reference "AI is artificial intelligence"')
    
    print("\n2. Environment variables:")
    print('   export OCI_COMPARTMENT_ID="ocid1.compartment..."')
    print('   export OCI_CONFIG_PROFILE="MY_PROFILE"')
    print('   python accuracy_checker.py --model grok3 --prompt "What is AI?" --reference "AI reference"')
    
    print("\n3. Interactive mode:")
    print('   python accuracy_checker.py --compartment-id "ocid1.compartment..." --model grok3')
    
    print("\n4. Different models:")
    print('   python accuracy_checker.py --compartment-id "ocid1.compartment..." --model llama4 --prompt "Question" --reference "Answer"')
    print('   python accuracy_checker.py --compartment-id "ocid1.compartment..." --model cohere --prompt "Question" --reference "Answer"')
    print('   python accuracy_checker.py --compartment-id "ocid1.compartment..." --model grok3 --prompt "Question" --reference "Answer"')
    
    print("\n5. List available models:")
    print('   python accuracy_checker.py --compartment-id "ocid1.compartment..." --list-models')
    
    print("\n" + "="*80)
    print("🔧 REQUIRED CONFIGURATION")
    print("="*80)
    print("• Compartment ID: --compartment-id or OCI_COMPARTMENT_ID env var")
    print("• OCI Config: ~/.oci/config (or custom path via --config-file)")
    print("• Valid OCI credentials with GenAI service access")
    print("• Python packages: oci")
    
    print("\n" + "="*80)
    print("🤖 AVAILABLE MODELS")
    print("="*80)
    print("• llama4: Llama 4 (GENERIC API format)")
    print("• cohere: Cohere (COHERE API format)")
    print("• grok3: Grok 3 (GENERIC API format)")
    print("• llama33-70b: Llama 3.3 70B (GENERIC API format)")
    
    print("\n" + "="*80)
    print("🌍 ENVIRONMENT VARIABLES")
    print("="*80)
    print("• OCI_COMPARTMENT_ID: OCI Compartment OCID")
    print("• OCI_CONFIG_PROFILE: Profile name in config file (default: DEFAULT)")
    print("• OCI_CONFIG_FILE: Path to OCI config file (default: ~/.oci/config)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='OCI Generative AI Comprehensive Assessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="For detailed usage examples, run: python accuracy_checker.py --help-examples"
    )
    parser.add_argument('--prompt', '-p', type=str, help='Input prompt for the AI model')
    parser.add_argument('--reference', '-ref', type=str, help='Reference answer for comparison')
    parser.add_argument('--model', '-m', type=str, 
                       default="llama4",
                       help='Model to use: llama4, cohere, grok3, llama33-70b, or full model OCID')
    parser.add_argument('--compartment-id', '-c', type=str, 
                       help='OCI Compartment ID (required - can also use OCI_COMPARTMENT_ID env var)')
    parser.add_argument('--max-tokens', '-t', type=int, help='Maximum tokens (will use model default if not specified)')
    parser.add_argument('--temperature', '-temp', type=float, default=0.7, help='Temperature')
    parser.add_argument('--config-profile', '-cp', type=str, default="DEFAULT", 
                       help='OCI Config Profile (can also use OCI_CONFIG_PROFILE env var)')
    parser.add_argument('--config-file', '-cf', type=str, 
                       help='OCI Config File path (default: ~/.oci/config, can also use OCI_CONFIG_FILE env var)')
    parser.add_argument('--list-models', action='store_true', help='List available model configurations')
    parser.add_argument('--help-examples', action='store_true', help='Show detailed usage examples and configuration help')
    
    args = parser.parse_args()
    
    # Show help examples if requested
    if args.help_examples:
        print_usage_examples()
        return
    
    # Get configuration from command line arguments or environment variables
    compartment_id = args.compartment_id or os.getenv('OCI_COMPARTMENT_ID')
    config_profile = args.config_profile or os.getenv('OCI_CONFIG_PROFILE', 'DEFAULT')
    config_file = args.config_file or os.getenv('OCI_CONFIG_FILE')
    
    # Validate required parameters
    if not compartment_id:
        print("❌ Error: Compartment ID is required!")
        print("   Provide it via --compartment-id argument or OCI_COMPARTMENT_ID environment variable")
        return
    
    # Initialize assessor
    try:
        assessor = OCIGenAIAssessor(
            compartment_id=compartment_id,
            config_profile=config_profile,
            config_file=config_file
        )
    except Exception as e:
        print(f"❌ Error initializing OCI client: {str(e)}")
        print("   Please check your OCI configuration and credentials")
        return
    
    # List models if requested
    if args.list_models:
        print("\n Available Model Configurations:")
        for key, config in assessor.model_configs.items():
            print(f"    {key}: {config['name']} ({config['api_format']}, max_tokens: {config['max_tokens']})")
        return
    
    # Get model configuration
    model_config = assessor.get_model_config(args.model)
    
    if args.prompt and args.reference:
        prompt = args.prompt
        reference = args.reference
    else:
        print("\n=== OCI Generative AI Comprehensive Assessor ===")
        print("This tool will generate AI responses and assess them for:")
        print(" Factual Accuracy (60%)")
        print(" Readability (20%)")
        print("⚡ Speed (20%)")
        
        print(f"\nUsing Model: {model_config['name']} ({model_config['api_format']})")
        print(f"Max Tokens: {model_config['max_tokens']}")
        print(f"Compartment: {compartment_id}")
        print(f"Config Profile: {config_profile}")
        if config_file:
            print(f"Config File: {config_file}")
        
        prompt = input("\nEnter your prompt: ").strip()
        reference = input("Enter the reference answer: ").strip()
    
    if not prompt or not reference:
        print(" Both prompt and reference answer are required!")
        return
    
    # Run comprehensive assessment
    try:
        results = assessor.comprehensive_assessment(
            prompt=prompt,
            reference_answer=reference,
            model_key=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"assessment_results/oci_genai_assessment_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n Results saved to: {filename}")
        
    except Exception as e:
        print(f" Error during assessment: {str(e)}")

if __name__ == "__main__":
    main()
