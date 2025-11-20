"""
Wrapper for llama-cpp-python to be compatible with the existing AnomalyGPT architecture
This allows using GGUF models on CPU without CUDA
"""
import torch
import torch.nn as nn
from llama_cpp import Llama
import numpy as np


class LlamaCppWrapper(nn.Module):
    """Wrapper to make llama-cpp-python compatible with PyTorch model interface"""
    
    def __init__(self, model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0):
        super().__init__()
        print(f"Loading GGUF model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,  # 0 for CPU-only
            verbose=False
        )
        
        # Mock config to match LlamaForCausalLM interface
        self.config = type('obj', (object,), {
            'hidden_size': 4096,  # Standard for 7B models
            'vocab_size': 32000,
            'pad_token_id': 0,
        })()
        
        self.device_type = 'cpu'
        
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_p=0.9, 
                 stopping_criteria=None, **kwargs):
        """Generate text given input token IDs"""
        # Convert input_ids to text (this is a simplified approach)
        # In practice, we'd need the tokenizer to decode properly
        # For now, we'll work with the prompt directly in the higher-level function
        
        # Return dummy tensor for compatibility
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        generated_len = min(max_new_tokens, 50)  # Limit for now
        
        # Return extended input_ids (dummy generation)
        output = torch.cat([
            input_ids,
            torch.zeros((batch_size, generated_len), dtype=torch.long)
        ], dim=1)
        
        return output
    
    def generate_text(self, prompt, max_tokens=128, temperature=0.7, top_p=0.9, stop=None):
        """Generate text from a text prompt (more suitable for llama-cpp-python)"""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop if stop else ["###", "\n\n"],
            echo=False
        )
        
        return response['choices'][0]['text']
    
    def __call__(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass - returns dummy outputs for compatibility"""
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        
        # Return mock output structure similar to LlamaForCausalLM
        class Output:
            def __init__(self):
                self.loss = None
                self.logits = torch.randn(batch_size, seq_len, vocab_size)
                
        return Output()
    
    def to(self, device):
        """Mock to() method for compatibility"""
        return self
    
    def eval(self):
        """Mock eval() method"""
        return self
    
    def train(self, mode=True):
        """Mock train() method"""
        return self
    
    def parameters(self):
        """Return empty parameters (GGUF model is not trainable in this setup)"""
        return []
    
    def named_parameters(self):
        """Return empty named parameters"""
        return []
    
    def state_dict(self):
        """Return empty state dict"""
        return {}
    
    def load_state_dict(self, state_dict, strict=True):
        """Mock load_state_dict - GGUF model is already loaded"""
        print("Note: GGUF model doesn't support loading additional state_dict")
        pass


class SimpleLlamaTokenizer:
    """Simple tokenizer wrapper for llama-cpp-python"""
    
    def __init__(self, llm):
        self.llm = llm
        self.pad_token = "</s>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "right"
        
    def __call__(self, text, add_special_tokens=True, return_tensors=None, **kwargs):
        """Tokenize text"""
        if isinstance(text, list):
            # Batch encoding
            tokens = [self.llm.tokenize(t.encode('utf-8')) for t in text]
            max_len = max(len(t) for t in tokens)
            
            # Pad sequences
            padded = []
            for t in tokens:
                padded.append(t + [self.pad_token_id] * (max_len - len(t)))
            
            result = {
                'input_ids': torch.tensor(padded, dtype=torch.long)
            }
        else:
            tokens = self.llm.tokenize(text.encode('utf-8'))
            result = {
                'input_ids': torch.tensor([tokens], dtype=torch.long)
            }
        
        if return_tensors == 'pt':
            return result
        return result
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids[0], list):
            # Batch decoding
            return [self.llm.detokenize(ids).decode('utf-8', errors='ignore') for ids in token_ids]
        else:
            return self.llm.detokenize(token_ids).decode('utf-8', errors='ignore')
    
    def batch_decode(self, token_ids, skip_special_tokens=True):
        """Batch decode token IDs"""
        return self.decode(token_ids, skip_special_tokens)
