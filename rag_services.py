# backend/services/rag_services.py
import os
import torch
import numpy as np
import faiss
import json
import logging
from typing import List, Dict

# More robust import handling
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Transformers library not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "transformers"])
    from transformers import AutoTokenizer, AutoModel

# Try different embedding approaches
class EmbeddingService:
    def __init__(self, model_name='bert-base-uncased'):
        """
        Flexible embedding service with multiple fallback mechanisms
        
        Args:
            model_name (str): Hugging Face model name for embeddings
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load embedding model with comprehensive error handling
        """
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Attempting alternative embedding strategy...")
            self._fallback_embedding_strategy()
    
    def _fallback_embedding_strategy(self):
        """
        Fallback embedding strategy if primary model fails
        """
        alternative_models = [
            'distilbert-base-uncased',
            'bert-base-uncased',
            'roberta-base'
        ]
        
        for alt_model in alternative_models:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(alt_model)
                self.model = AutoModel.from_pretrained(alt_model).to(self.device)
                self.model.eval()
                self.model_name = alt_model
                print(f"Fallback to alternative model: {alt_model}")
                return
            except Exception as e:
                print(f"Failed to load {alt_model}: {e}")
        
        raise RuntimeError("Could not load any embedding model")
    
    def encode(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Generate embeddings for input texts
        
        Args:
            texts (List[str]): List of texts to embed
            max_length (int): Maximum sequence length
        
        Returns:
            Numpy array of embeddings
        """
        try:
            # Tokenize texts
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Mean pooling
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()
        
        except Exception as e:
            print(f"Embedding generation error: {e}")
            # Basic fallback: return zero vector
            return np.zeros((len(texts), self.model.config.hidden_size))

class MeetingRAGService:
    def __init__(self, 
                 index_path='data/faiss_index.bin',
                 meeting_data_path='data/meeting_data.json'):
        """
        Initialize RAG service with robust error handling
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Paths for storage
        self.index_path = index_path
        self.meeting_data_path = meeting_data_path
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        
        # Create data directory if not exists
        os.makedirs('data', exist_ok=True)
        
        # Initialize FAISS index
        self.index = self._load_or_create_index()
        
        # Load meeting data
        self.meeting_data = self._load_meeting_data()
    
    def _load_or_create_index(self):
        """Create or load FAISS index"""
        try:
            if os.path.exists(self.index_path):
                return faiss.read_index(self.index_path)
            
            # Create index with dimensionality from embedding model
            embedding_dim = self.embedding_service.model.config.hidden_size
            index = faiss.IndexFlatL2(embedding_dim)
            return index
        except Exception as e:
            self.logger.error(f"Index loading error: {e}")
            raise
    
    def _load_meeting_data(self):
        """Load existing meeting data"""
        try:
            if os.path.exists(self.meeting_data_path):
                with open(self.meeting_data_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Meeting data loading error: {e}")
            return {}
    
    def add_meeting_to_index(self, meeting_data: Dict):
        """Add meeting to RAG index"""
        try:
            # Prepare text for embedding
            text_to_embed = self._prepare_embedding_text(meeting_data)
            
            # Generate embedding
            embedding = self.embedding_service.encode([text_to_embed])[0]
            
            # Add to index
            self.index.add(np.array([embedding]))
            
            # Save meeting data
            meeting_id = meeting_data.get('meeting_id', f'meeting_{len(self.meeting_data)}')
            self.meeting_data[meeting_id] = meeting_data
            
            # Persist data
            self._save_index_and_data()
            
            self.logger.info(f"Added meeting {meeting_id} to index")
        except Exception as e:
            self.logger.error(f"Error adding meeting: {e}")
    
    def _save_index_and_data(self):
        """Save FAISS index and meeting data"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.meeting_data_path, 'w') as f:
                json.dump(self.meeting_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving index and data: {e}")
    
    def _prepare_embedding_text(self, meeting_data: Dict) -> str:
        """Prepare text for embedding"""
        return ' '.join([
            meeting_data.get('summary', {}).get('summary', ''),
            ', '.join(meeting_data.get('summary', {}).get('discussion_topics', [])),
            ', '.join(meeting_data.get('crm_insights', {}).get('pain_points', [])),
            ', '.join(meeting_data.get('crm_insights', {}).get('action_items', []))
        ])
    
    def search_past_meetings(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search past meetings"""
        try:
            # Embed query
            query_embedding = self.embedding_service.encode([query])[0]
            
            # Perform search
            distances, indices = self.index.search(
                np.array([query_embedding]), 
                k=min(top_k, self.index.ntotal)
            )
            
            # Collect results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    meeting_info = list(self.meeting_data.values())[idx]
                    results.append({
                        "similarity_score": 1 / (1 + dist),
                        "content": self._format_meeting_result(meeting_info)
                    })
            
            return results
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def _format_meeting_result(self, meeting_info: Dict) -> str:
        """Format meeting result for display"""
        return f"""
        Meeting Summary:
        {meeting_info.get('summary', {}).get('summary', 'No summary')}

        Discussion Topics:
        {', '.join(meeting_info.get('summary', {}).get('discussion_topics', []))}

        Action Items:
        {', '.join(meeting_info.get('crm_insights', {}).get('action_items', []))}
        """

# Global service instances
rag_service = MeetingRAGService()

def create_and_add_to_faiss(meeting_data: Dict):
    """Wrapper to add meeting to index"""
    rag_service.add_meeting_to_index(meeting_data)

def search_faiss(query: str, top_k: int = 3):
    """Wrapper to search past meetings"""
    return rag_service.search_past_meetings(query, top_k)

# Test functionality when script is run directly
if __name__ == "__main__":
    # Example usage
    test_meeting_data = {
        'meeting_id': 'test_meeting_1',
        'summary': {
            'summary': 'A test meeting about project planning',
            'discussion_topics': ['Project Scope', 'Timeline', 'Resources']
        },
        'crm_insights': {
            'pain_points': ['Limited budget', 'Tight deadline'],
            'action_items': ['Finalize project plan', 'Schedule follow-up meeting']
        }
    }
    
    # Add meeting to index
    create_and_add_to_faiss(test_meeting_data)
    
    # Search meetings
    results = search_faiss("project planning")
    print("Search Results:")
    for result in results:
        print(result)